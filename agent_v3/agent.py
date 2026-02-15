"""AgentV3: Gemini Robotics ER 1.5 robot arm control orchestrator."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
from google import genai
from google.genai import types

from agent.camera import RealSenseCamera
from agent.vlm_agent import ConfirmationHandler
from motion_controller.motion import Motion

from agent_v2.calibration import DEFAULT_CALIBRATION_PATH
from agent_v2.coordinate_transform import CoordinateTransform

from .gemini_vision import GeminiVision
from .prompts import SYSTEM_PROMPT, create_task_prompt
from .tools import create_gemini_tools

logger = logging.getLogger("agent_v3")

# Gripper defaults (for direct SDK calls — Motion doesn't wrap gripper)
GRIPPER_SPEED = 100
GRIPPER_ACC = 50


class AgentV3:
    """Gemini Robotics ER-driven robot arm control agent.

    Replaces both GPT-5.2 and SAM2 from AgentV2 with a single Gemini
    Robotics ER 1.5 model that handles reasoning, tool-calling, and
    spatial vision (detection, pointing, segmentation) natively.
    """

    def __init__(
        self,
        google_api_key: str,
        model: str = "gemini-robotics-er-1.5-preview",
        arm_port: str | None = None,
        auto_confirm: bool = False,
        debug: bool = False,
        thinking_budget: int = 1024,
        calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    ):
        self.debug = debug
        self.thinking_budget = thinking_budget

        if debug:
            logging.getLogger("agent_v3").setLevel(logging.DEBUG)

        logger.info(f"Initializing AgentV3 with model={model}")

        # Gemini client
        self.client = genai.Client(api_key=google_api_key)
        self.model = model

        # Vision wrapper (uses same client)
        self.vision = GeminiVision(self.client, model=model)

        # Hardware
        self.camera = RealSenseCamera()
        self.motion = Motion(port=arm_port, inverted=True)

        # Coordinate transform
        self.ct = CoordinateTransform(calibration_path=calibration_path)
        self._calibration_path = calibration_path

        # Confirmation
        self.confirmation = ConfirmationHandler(auto_confirm=auto_confirm)

        # Tools
        self.tools = create_gemini_tools()

        # Frame cache (populated by look(), used by detect() and goto_pixel())
        self._last_color_image: np.ndarray | None = None
        self._last_depth_frame: rs.depth_frame | None = None
        self._last_depth_image: np.ndarray | None = None

    def start(self) -> None:
        """Start the agent (initialize hardware)."""
        print("Starting camera...")
        self.camera.start()

        # Set up coordinate transform intrinsics from camera
        self.ct.set_intrinsics_from_camera(self.camera)

        # Probe ground (required before any Motion movement)
        print("Probing ground level...")
        self.motion.probe_ground()

        print("Moving arm to home position...")
        self.motion.home()
        time.sleep(1)
        print("Agent ready.")

    def stop(self) -> None:
        """Stop the agent (cleanup hardware)."""
        print("\nStopping agent...")
        self.motion.home()
        time.sleep(1)
        self.camera.stop()
        print("Agent stopped.")

    # ── Frame capture ──────────────────────────────────────────

    def _capture_aligned_frames(self) -> tuple[np.ndarray, np.ndarray, rs.depth_frame]:
        """Capture aligned color + depth frames.

        Returns:
            color_image: BGR numpy array
            depth_colorized: Colorized depth numpy array
            depth_frame: Raw RealSense depth frame (for coordinate queries)
        """
        if not self.camera._started:
            self.camera.start()

        frames = self.camera.pipeline.wait_for_frames()

        # Align depth to color
        aligned = self.ct.get_aligned_frames(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")

        color_image = np.asanyarray(color_frame.get_data())

        # Colorize depth for visualization
        colorized = self.camera.colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized.get_data())
        depth_image = self.camera._draw_depth_scale(
            depth_image.copy(), self.camera.min_depth_m, self.camera.max_depth_m
        )

        return color_image, depth_image, depth_frame

    def _encode_image_for_gemini(self, image: np.ndarray) -> types.Part:
        """Encode a BGR numpy image as a JPEG Part for Gemini messages."""
        _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return types.Part.from_bytes(data=jpeg.tobytes(), mime_type="image/jpeg")

    @staticmethod
    def _draw_coordinate_grid(image: np.ndarray) -> np.ndarray:
        """Draw a labeled coordinate grid every 80px on a copy of the image."""
        out = image.copy()
        h, w = out.shape[:2]
        color = (180, 180, 180)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1

        for x in range(0, w, 80):
            cv2.line(out, (x, 0), (x, h), color, 1)
            cv2.putText(out, str(x), (x + 2, 12), font, font_scale, color, thickness)

        for y in range(0, h, 80):
            cv2.line(out, (0, y), (w, y), color, 1)
            cv2.putText(out, str(y), (2, y + 12), font, font_scale, color, thickness)

        return out

    # ── Tool executors ──────────────────────────────────────────

    def _execute_look(self, _args: dict[str, Any]) -> tuple[str, list[types.Part]]:
        """Execute look tool: capture frame, return images as Gemini Parts.

        Returns (text_result, image_parts).
        """
        color_image, depth_image, depth_frame = self._capture_aligned_frames()

        # Cache for subsequent detect() / goto_pixel() calls
        self._last_color_image = color_image
        self._last_depth_frame = depth_frame
        self._last_depth_image = depth_image

        h, w = color_image.shape[:2]
        text_result = f"Camera frame captured ({w}x{h}). Ready for detect() queries."

        # Draw coordinate grid on a copy
        color_with_grid = self._draw_coordinate_grid(color_image)
        color_part = self._encode_image_for_gemini(color_with_grid)
        depth_part = self._encode_image_for_gemini(depth_image)

        image_parts = [
            color_part,
            types.Part.from_text(
                text="[Camera image with coordinate grid (labels every 80px). Depth image below.]"
            ),
            depth_part,
        ]

        return text_result, image_parts

    def _execute_detect(self, args: dict[str, Any]) -> tuple[str, list[types.Part]]:
        """Execute detect tool: Gemini ER segmentation with natural language query.

        Returns (text_result, image_parts).
        """
        query = args["query"]

        if self._last_color_image is None or self._last_depth_frame is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No image available. Call look() first.",
                }
            ), []

        # Run Gemini ER segmentation
        print(f"Detecting: {query!r}...")
        segments = self.vision.segment(self._last_color_image, query)

        if not segments:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"No objects matching '{query}' detected.",
                }
            ), []

        # Use the first (best) result
        seg = segments[0]
        bbox_px = seg.get("box_2d_px", (0, 0, 0, 0))
        label = seg.get("label", query)
        centroid = seg.get(
            "centroid", ((bbox_px[0] + bbox_px[2]) // 2, (bbox_px[1] + bbox_px[3]) // 2)
        )
        cx, cy = centroid

        # Compute 3D arm coordinates from centroid
        arm_coords = None
        depth_mm = self.ct.get_depth_at_pixel(self._last_depth_frame, cx, cy)
        if depth_mm is not None:
            cam_3d = self.ct.deproject_pixel(cx, cy, depth_mm=depth_mm)
            if cam_3d is not None:
                arm_3d = self.ct.camera_to_arm(cam_3d)
                if arm_3d is not None:
                    arm_coords = {
                        "x": round(float(arm_3d[0]), 1),
                        "y": round(float(arm_3d[1]), 1),
                        "z": round(float(arm_3d[2]), 1),
                    }

        # Get mask if available
        mask = seg.get("mask_full")
        mask_area = int(mask.sum()) if mask is not None else 0

        result = {
            "status": "success",
            "query": query,
            "label": label,
            "centroid": {"pixel_x": cx, "pixel_y": cy},
            "bbox": {
                "x1": bbox_px[0],
                "y1": bbox_px[1],
                "x2": bbox_px[2],
                "y2": bbox_px[3],
            },
            "mask_area_px": mask_area,
            "depth_mm": round(depth_mm, 1) if depth_mm else None,
            "arm_coordinates": arm_coords,
        }

        text_result = json.dumps(result)

        # Draw annotation overlay
        annotated = self._annotate_detection(
            self._last_color_image, mask, bbox_px, (cx, cy), label
        )
        annotated_part = self._encode_image_for_gemini(annotated)

        image_parts = [
            annotated_part,
            types.Part.from_text(
                text=f"[Detection result for '{query}'. Label: {label}, "
                f"Centroid: ({cx},{cy}), Mask area: {mask_area}px]"
            ),
        ]

        return text_result, image_parts

    def _annotate_detection(
        self,
        color_image: np.ndarray,
        mask: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        centroid: tuple[int, int],
        label: str,
    ) -> np.ndarray:
        """Draw detection annotation on the image."""
        annotated = color_image.copy()

        # Green mask overlay (semi-transparent) if mask is available
        if mask is not None and mask.any():
            green_overlay = annotated.copy()
            green_overlay[mask] = [0, 200, 0]
            cv2.addWeighted(green_overlay, 0.4, annotated, 0.6, 0, annotated)

            # Contour outline
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

        # Bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Label
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # Centroid dot (blue)
        mx, my = centroid
        cv2.circle(annotated, (mx, my), 6, (255, 0, 0), -1)
        cv2.circle(annotated, (mx, my), 6, (255, 255, 255), 1)

        # Coordinate grid
        annotated = self._draw_coordinate_grid(annotated)

        return annotated

    def _execute_goto_pixel(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute goto_pixel tool: pixel -> depth -> deproject -> affine -> move."""
        pixel_x = args["pixel_x"]
        pixel_y = args["pixel_y"]
        z_offset_mm = args.get("z_offset_mm", 50)

        if self._last_depth_frame is None:
            return {
                "status": "error",
                "message": "No depth frame available. Call look() first.",
            }

        # Get depth at pixel
        depth_mm = self.ct.get_depth_at_pixel(self._last_depth_frame, pixel_x, pixel_y)
        if depth_mm is None:
            return {
                "status": "error",
                "message": f"No depth at pixel ({pixel_x}, {pixel_y}) — surface may be too close/far.",
            }

        # Deproject to camera 3D
        cam_3d = self.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
        if cam_3d is None:
            return {
                "status": "error",
                "message": f"Deprojection failed at ({pixel_x}, {pixel_y}).",
            }

        # Transform to arm coordinates
        arm_3d = self.ct.camera_to_arm(cam_3d)
        if arm_3d is None:
            return {
                "status": "error",
                "message": "Camera-to-arm transform failed. Calibration may be missing.",
            }

        target_x = float(arm_3d[0])
        target_y = float(arm_3d[1])
        target_z = float(arm_3d[2]) + z_offset_mm

        ok = self.motion.move_to(target_x, target_y, target_z)
        if ok:
            return {
                "status": "success",
                "message": (
                    f"Moved to pixel ({pixel_x},{pixel_y}) -> "
                    f"arm x={target_x:.0f}, y={target_y:.0f}, z={target_z:.0f}"
                ),
                "pixel": {"x": pixel_x, "y": pixel_y},
                "arm_position": {
                    "x": round(target_x, 1),
                    "y": round(target_y, 1),
                    "z": round(target_z, 1),
                },
            }
        else:
            return {
                "status": "error",
                "message": (
                    f"move_to failed — position ({target_x:.0f}, {target_y:.0f}, {target_z:.0f}) "
                    "may be unreachable"
                ),
            }

    def _execute_pose_get(self, _args: dict[str, Any]) -> dict[str, Any]:
        x, y, z = self.motion.get_pose()
        return {
            "status": "success",
            "position": {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1)},
            "message": f"Current position: x={x:.1f}, y={y:.1f}, z={z:.1f} (ground-relative)",
        }

    def _execute_move_home(self, _args: dict[str, Any]) -> dict[str, Any]:
        self.motion.home()
        time.sleep(1)
        return {"status": "success", "message": "Moved to home position"}

    def _execute_gripper_ctrl(self, args: dict[str, Any]) -> dict[str, Any]:
        angle = max(0, min(90, args["angle"]))
        self.motion.arm.gripper_angle_ctrl(
            angle=angle, speed=GRIPPER_SPEED, acc=GRIPPER_ACC
        )
        time.sleep(0.5)
        state = "open" if angle > 45 else "closed"
        return {
            "status": "success",
            "gripper_angle": angle,
            "message": f"Gripper {state} at {angle} degrees",
        }

    def _execute_tool(
        self, name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any] | str, list[types.Part] | None]:
        """Execute a tool call. Returns (result, optional_image_parts)."""
        # look and detect return images, no confirmation needed
        if name == "look":
            print("Capturing camera frame...")
            text_result, images = self._execute_look(args)
            return text_result, images

        if name == "detect":
            print(f"Detecting: {args.get('query', '?')}...")
            text_result, images = self._execute_detect(args)
            return text_result, images

        # Confirmation for movement tools
        needs_confirm = name in ["goto_pixel", "move_home", "gripper_ctrl"]
        if needs_confirm and not self.confirmation.auto_confirm:
            if name == "goto_pixel":
                px, py = args["pixel_x"], args["pixel_y"]
                z_off = args.get("z_offset_mm", 50)
                arm_desc = ""
                if self._last_depth_frame is not None:
                    depth_mm = self.ct.get_depth_at_pixel(
                        self._last_depth_frame, px, py
                    )
                    if depth_mm is not None:
                        cam_3d = self.ct.deproject_pixel(px, py, depth_mm=depth_mm)
                        if cam_3d is not None:
                            arm_3d = self.ct.camera_to_arm(cam_3d)
                            if arm_3d is not None:
                                arm_desc = (
                                    f" -> arm x={arm_3d[0]:.0f}, y={arm_3d[1]:.0f}, "
                                    f"z={float(arm_3d[2]) + z_off:.0f}"
                                )
                action_str = f"GOTO pixel ({px},{py}) z_offset={z_off}mm{arm_desc}"
            elif name == "move_home":
                action_str = "MOVE ARM to HOME position"
            elif name == "gripper_ctrl":
                state = "OPEN" if args["angle"] > 45 else "CLOSE"
                action_str = f"GRIPPER {state} to {args['angle']:.1f} degrees"
            else:
                action_str = f"Execute {name} with args: {json.dumps(args)}"

            print("\n" + "=" * 50)
            print("PENDING ACTION:")
            print(action_str)
            print("=" * 50)

            while True:
                response = input("Execute this action? [y/n/q]: ").strip().lower()
                if response in ["y", "yes"]:
                    break
                elif response in ["n", "no"]:
                    return {"status": "cancelled", "message": "User cancelled"}, None
                elif response in ["q", "quit"]:
                    raise KeyboardInterrupt("User requested quit")
                else:
                    print("Please enter 'y', 'n', or 'q'")

        executors = {
            "goto_pixel": self._execute_goto_pixel,
            "pose_get": self._execute_pose_get,
            "move_home": self._execute_move_home,
            "gripper_ctrl": self._execute_gripper_ctrl,
        }

        executor = executors.get(name)
        if executor is None:
            return {"status": "error", "message": f"Unknown tool: {name}"}, None

        try:
            print(f"Executing {name}...")
            result = executor(args)
            return result, None
        except Exception as e:
            return {"status": "error", "message": str(e)}, None

    # ── Agent loop ──────────────────────────────────────────────

    def process_task(self, task: str) -> str:
        """Process a task using the Gemini ER agent loop."""
        logger.info(f"Processing task: {task}")

        task_prompt = create_task_prompt(task)

        # Gemini has no "system" role — inject system prompt via initial exchange
        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=SYSTEM_PROMPT)],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Understood. I'm ready to control the robot arm using my vision "
                        "and tool-calling capabilities. Give me a task and I'll start by "
                        "looking at the scene."
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=task_prompt)],
            ),
        ]

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Agent iteration {iteration} ---")

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=self.tools,
                        temperature=0.3,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        ),
                    ),
                )
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise

            if not response.candidates:
                logger.warning("No candidates in response")
                return "Model returned empty response."

            candidate = response.candidates[0]
            model_parts = (candidate.content and candidate.content.parts) or []

            if not model_parts:
                logger.warning("Empty parts in response")
                return "Model returned empty response."

            # Collect text output
            text_output = ""
            for part in model_parts:
                if part.text:
                    text_output += part.text

            if text_output:
                print(f"\nAgent: {text_output}")

            # Add model response to conversation
            contents.append(candidate.content)

            # Check for function calls
            function_calls = [p for p in model_parts if p.function_call]

            if not function_calls:
                # No tool calls — model is done
                return text_output or "Task completed."

            # Execute tool calls
            movement_executed = False
            pending_image_parts: list[types.Part] = []
            function_response_parts: list[types.Part] = []

            for part in function_calls:
                fc = part.function_call
                name = fc.name
                args = dict(fc.args) if fc.args else {}

                result, images = self._execute_tool(name, args)
                result_str = result if isinstance(result, str) else json.dumps(result)
                print(f"Result: {result_str[:200]}")

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result_str},
                    )
                )

                if images:
                    pending_image_parts.extend(images)

                if name in ["goto_pixel", "move_home", "gripper_ctrl"]:
                    result_dict = result if isinstance(result, dict) else {}
                    if result_dict.get("status") == "success":
                        movement_executed = True

            # Add function responses as a user turn
            contents.append(types.Content(role="user", parts=function_response_parts))

            # Add images from look/detect as a separate user message
            if pending_image_parts:
                contents.append(types.Content(role="user", parts=pending_image_parts))

            # After movement, capture a fresh view for verification
            if movement_executed:
                print("Capturing post-movement view...")
                time.sleep(0.5)
                text_result, image_parts = self._execute_look({})

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            *image_parts,
                            types.Part.from_text(
                                text="[Updated view after movement. "
                                "Confirm task completion or continue if more actions needed.]"
                            ),
                        ],
                    )
                )

        return "Maximum iterations reached. Task may be incomplete."

    # ── Interactive REPL ────────────────────────────────────────

    def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        self.start()

        print("\n" + "=" * 60)
        print("Agent V3 - Gemini Robotics ER Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  quit       - Exit the agent")
        print("  home       - Move arm to home position")
        print("  pos        - Show current arm position (ground-relative)")
        print("  calibrate  - Run calibration procedure")
        print("  touch      - Touch debug (click to move arm, then reset)")
        print()
        print("Or enter a task like: 'touch the red cup'")
        print()

        try:
            while True:
                task = input("Task> ").strip()
                if not task:
                    continue

                if task.lower() in ["quit", "exit", "q"]:
                    break

                if task.lower() == "home":
                    self.motion.home()
                    time.sleep(1)
                    print("Moved to home position.")
                    continue

                if task.lower() == "pos":
                    x, y, z = self.motion.get_pose()
                    print(
                        f"Current position: x={x:.1f}, y={y:.1f}, z={z:.1f} (ground-relative)"
                    )
                    continue

                if task.lower() == "calibrate":
                    self._run_calibration()
                    continue

                if task.lower() == "touch":
                    self._run_touch_debug()
                    continue

                print("\nProcessing task...")
                try:
                    self.process_task(task)
                    print(f"\n{'=' * 60}")
                    print("Task completed.")
                    print(f"{'=' * 60}\n")
                except Exception as e:
                    if self.debug:
                        import traceback

                        traceback.print_exc()
                    print(f"\nError processing task: {e}\n")

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.stop()

    def _run_calibration(self) -> None:
        """Run the interactive calibration procedure."""
        from agent_v2.calibration import CalibrationProcedure

        try:
            proc = CalibrationProcedure(self.camera, self.motion, self.ct)
            R, t, rmse = proc.run(save_path=self._calibration_path)
            self.ct._M = np.column_stack([R, t])
            print("Calibration loaded into coordinate transform.")
        except Exception as e:
            print(f"Calibration failed: {e}")

    def _run_touch_debug(self) -> None:
        """Interactive touch debug: click a point, arm moves there, then resets.

        Shows side-by-side color + depth. Click on either image to pick a
        target. The pixel is deprojected to 3D, transformed to arm
        coordinates, and the arm moves there. After a pause, the arm
        returns home and waits for the next click. Press 'q' to exit.
        """
        if not self.ct.has_calibration:
            print(
                "ERROR: Touch debug requires calibration.\n"
                "  Run 'calibrate' first, or start with --calibrate."
            )
            return

        window_name = "Touch Debug - Click to move arm (Q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        clicked_pixel: list[tuple[int, int] | None] = [None]
        color_width = [0]

        def mouse_callback(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if color_width[0] > 0 and x >= color_width[0]:
                    x = x - color_width[0]
                clicked_pixel[0] = (x, y)

        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "=" * 60)
        print("TOUCH DEBUG MODE")
        print("=" * 60)
        print("Click on either image to move the arm there.")
        print("The arm will move to the clicked 3D point, pause, then go home.")
        print("Press 'q' in the window to exit.\n")

        print("Moving arm to home position...")
        self.motion.home()
        time.sleep(1.0)

        try:
            while True:
                color_image, depth_image, depth_frame = self._capture_aligned_frames()
                color_width[0] = color_image.shape[1]

                color_display = color_image.copy()
                depth_display = depth_image.copy()

                cv2.putText(
                    color_display,
                    "Click to move arm | Q=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                x, y, z = self.motion.get_pose()
                cv2.putText(
                    color_display,
                    f"Arm: x={x:.0f} y={y:.0f} z={z:.0f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                if clicked_pixel[0]:
                    cx, cy = clicked_pixel[0]
                    for img in [color_display, depth_display]:
                        cv2.circle(img, (cx, cy), 8, (0, 0, 255), 2)
                        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
                        cv2.line(img, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)

                cv2.putText(
                    color_display,
                    "COLOR",
                    (10, color_display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    depth_display,
                    "DEPTH",
                    (10, depth_display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                combined = np.hstack([color_display, depth_display])
                cv2.imshow(window_name, combined)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    break

                if clicked_pixel[0] is None:
                    continue

                px, py = clicked_pixel[0]
                clicked_pixel[0] = None

                depth_mm = self.ct.get_depth_at_pixel(depth_frame, px, py)
                if depth_mm is None:
                    print(f"  No depth at pixel ({px}, {py}) — skipping.")
                    continue

                cam_3d = self.ct.deproject_pixel(px, py, depth_mm=depth_mm)
                if cam_3d is None:
                    print(f"  Deprojection failed at ({px}, {py}) — skipping.")
                    continue

                arm_3d = self.ct.camera_to_arm(cam_3d)
                if arm_3d is None:
                    print("  Camera->arm transform failed — skipping.")
                    continue

                target_x, target_y, target_z = (
                    float(arm_3d[0]),
                    float(arm_3d[1]),
                    float(arm_3d[2]),
                )

                print(
                    f"\n  Click: pixel=({px},{py}) depth={depth_mm:.0f}mm\n"
                    f"    cam_3d = [{cam_3d[0]:.1f}, {cam_3d[1]:.1f}, {cam_3d[2]:.1f}]\n"
                    f"    arm_3d = [{target_x:.1f}, {target_y:.1f}, {target_z:.1f}]"
                )

                ok = self.motion.move_to(target_x, target_y, target_z)
                if ok:
                    print("    Moved to target. Holding for 2 seconds...")
                    time.sleep(2.0)
                else:
                    print("    Move failed — position may be unreachable.")
                    time.sleep(0.5)

                print("    Returning home...")
                self.motion.home()
                time.sleep(1.0)

        finally:
            cv2.destroyAllWindows()
            print("Touch debug ended.")
