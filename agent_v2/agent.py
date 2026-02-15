"""AgentV2: Vision-driven robot arm control orchestrator using SAM2."""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

from agent.camera import RealSenseCamera
from agent.vlm_agent import ConfirmationHandler, create_openai_client
from motion_controller.motion import Motion

from .calibration import DEFAULT_CALIBRATION_PATH
from .coordinate_transform import CoordinateTransform
from .prompts import SYSTEM_PROMPT, create_task_prompt
from .tools import TOOL_DECLARATIONS
from .vision.sam2_backend import SAM2Backend

logger = logging.getLogger("agent_v2")

# Gripper defaults (for direct SDK calls — Motion doesn't wrap gripper)
GRIPPER_SPEED = 100
GRIPPER_ACC = 50


def convert_tools_to_openai_format() -> list[dict]:
    """Convert tool declarations to OpenAI function calling format."""
    tools = []
    for decl in TOOL_DECLARATIONS:
        tools.append({
            "type": "function",
            "function": {
                "name": decl["name"],
                "description": decl["description"],
                "parameters": decl["parameters"],
            },
        })
    return tools


class AgentV2:
    """Vision-driven robot arm control agent using SAM2.

    The LLM sees raw camera images and decides what to interact with.
    SAM2 provides precise segmentation masks at clicked points.
    The existing pixel-to-3D coordinate pipeline converts clicks to arm coords.

    Uses the Motion controller for arm movement (IK-based, ground-relative
    coordinates, torque-based ground probing).
    """

    @staticmethod
    def _draw_coordinate_grid(image: np.ndarray) -> np.ndarray:
        """Draw a labeled coordinate grid every 80px on a copy of the image."""
        out = image.copy()
        h, w = out.shape[:2]
        color = (180, 180, 180)  # light gray
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1

        # Draw vertical lines and top-edge labels
        for x in range(0, w, 80):
            cv2.line(out, (x, 0), (x, h), color, 1)
            cv2.putText(out, str(x), (x + 2, 12), font, font_scale, color, thickness)

        # Draw horizontal lines and left-edge labels
        for y in range(0, h, 80):
            cv2.line(out, (0, y), (w, y), color, 1)
            cv2.putText(out, str(y), (2, y + 12), font, font_scale, color, thickness)

        return out

    def __init__(
        self,
        openai_api_key: str,
        helicone_api_key: str,
        sam2_backend: SAM2Backend,
        model: str = "gpt-5.2",
        arm_port: str | None = None,
        auto_confirm: bool = False,
        debug: bool = False,
        reasoning_effort: str = "low",
        calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    ):
        self.debug = debug
        self.reasoning_effort = reasoning_effort

        if debug:
            logging.getLogger("agent_v2").setLevel(logging.DEBUG)

        logger.info(f"Initializing AgentV2 with model={model}, vision=sam2")

        # OpenAI client via Helicone
        self.client = create_openai_client(openai_api_key, helicone_api_key, debug=debug)
        self.model = model

        # Hardware
        self.camera = RealSenseCamera()
        self.motion = Motion(port=arm_port, inverted=True)

        # Vision
        self.sam2 = sam2_backend

        # Coordinate transform
        self.ct = CoordinateTransform(calibration_path=calibration_path)
        self._calibration_path = calibration_path

        # Confirmation
        self.confirmation = ConfirmationHandler(auto_confirm=auto_confirm)

        # Tools
        self.tools = convert_tools_to_openai_format()

        # Frame cache (populated by look(), used by segment() and goto_pixel())
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

    def _encode_image(self, image: np.ndarray, quality: int = 85) -> str:
        """Encode image as base64 JPEG."""
        _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(jpeg.tobytes()).decode("utf-8")

    # ── Tool executors ──────────────────────────────────────────

    def _execute_look(self, _args: dict[str, Any]) -> tuple[str, list[dict]]:
        """Execute look tool: capture frame, encode for SAM2, return images.

        Returns (text_result, image_content_blocks).
        """
        color_image, depth_image, depth_frame = self._capture_aligned_frames()

        # Cache for subsequent segment() / goto_pixel() calls
        self._last_color_image = color_image
        self._last_depth_frame = depth_frame
        self._last_depth_image = depth_image

        # Encode image for SAM2 (slow on CPU, ~2-5s)
        print("Encoding image for SAM2...")
        encode_time = self.sam2.set_image(color_image)

        h, w = color_image.shape[:2]
        text_result = (
            f"Camera frame captured ({w}x{h}). "
            f"SAM2 image encoded in {encode_time:.1f}s. "
            f"Ready for segment() queries."
        )

        # Draw coordinate grid on a copy (keep _last_color_image clean for SAM2)
        color_with_grid = self._draw_coordinate_grid(color_image)
        color_b64 = self._encode_image(color_with_grid)
        depth_b64 = self._encode_image(depth_image)

        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{color_b64}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": "[Camera image with coordinate grid (labels every 80px). Depth image below.]",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{depth_b64}",
                    "detail": "high",
                },
            },
        ]

        return text_result, image_content

    def _execute_segment(self, args: dict[str, Any]) -> tuple[str, list[dict]]:
        """Execute segment tool: SAM2 point-prompt segmentation.

        Returns (text_result, image_content_blocks).
        """
        pixel_x = args["pixel_x"]
        pixel_y = args["pixel_y"]

        if self._last_color_image is None or self._last_depth_frame is None:
            return json.dumps({
                "status": "error",
                "message": "No image available. Call look() first.",
            }), []

        # Run SAM2 segmentation
        print(f"Running SAM2 segmentation at ({pixel_x}, {pixel_y})...")
        t0 = time.time()
        mask, score, bbox = self.sam2.segment_point(pixel_x, pixel_y)
        seg_time = time.time() - t0
        logger.info(f"SAM2 segmentation in {seg_time:.2f}s, score={score:.3f}")

        # # Compute centroid (commented out — using exact click coordinates instead)
        # cx, cy = self.sam2.mask_centroid(mask)

        # Compute 3D arm coordinates from the exact click pixel
        arm_coords = None
        depth_mm = self.ct.get_depth_at_pixel(self._last_depth_frame, pixel_x, pixel_y)
        if depth_mm is not None:
            cam_3d = self.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
            if cam_3d is not None:
                arm_3d = self.ct.camera_to_arm(cam_3d)
                if arm_3d is not None:
                    arm_coords = {
                        "x": round(float(arm_3d[0]), 1),
                        "y": round(float(arm_3d[1]), 1),
                        "z": round(float(arm_3d[2]), 1),
                    }

        result = {
            "status": "success",
            "click": {"pixel_x": pixel_x, "pixel_y": pixel_y},
            "score": round(score, 3),
            "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
            "mask_area_px": int(mask.sum()),
            "depth_mm": round(depth_mm, 1) if depth_mm else None,
            "arm_coordinates": arm_coords,
            "segment_time_s": round(seg_time, 2),
        }

        text_result = json.dumps(result)

        # Draw annotation overlay
        annotated = self._annotate_segmentation(
            self._last_color_image, mask, bbox, (pixel_x, pixel_y)
        )
        annotated_b64 = self._encode_image(annotated)

        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{annotated_b64}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": (
                    f"[Segmentation mask overlay. Click target: ({pixel_x},{pixel_y}), "
                    f"Score: {score:.3f}]"
                ),
            },
        ]

        return text_result, image_content

    def _annotate_segmentation(
        self,
        color_image: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        click: tuple[int, int],
    ) -> np.ndarray:
        """Draw segmentation annotation on the image.

        - Green semi-transparent mask overlay
        - Green contour outline
        - Green bounding box
        - Red crosshair at click point (exact target location)
        """
        annotated = color_image.copy()

        # Green mask overlay (semi-transparent)
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

        # Click crosshair (red) — this is the exact target location
        cx, cy = click
        cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # # Centroid dot (blue) — commented out, using exact click coords instead
        # mx, my = centroid
        # cv2.circle(annotated, (mx, my), 6, (255, 0, 0), -1)
        # cv2.circle(annotated, (mx, my), 6, (255, 255, 255), 1)

        # Coordinate grid so the LLM can evaluate positions
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

    def _execute_tool(self, tool_call) -> tuple[dict[str, Any] | str, list[dict] | None]:
        """Execute a tool call. Returns (result, optional_image_content)."""
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

        # look and segment return images, no confirmation needed
        if tool_name == "look":
            print("Capturing camera frame...")
            text_result, images = self._execute_look(args)
            return text_result, images

        if tool_name == "segment":
            print(f"Segmenting at ({args.get('pixel_x')}, {args.get('pixel_y')})...")
            text_result, images = self._execute_segment(args)
            return text_result, images

        # Confirmation for movement tools
        needs_confirm = tool_name in ["goto_pixel", "move_home", "gripper_ctrl"]
        if needs_confirm and not self.confirmation.auto_confirm:
            if tool_name == "goto_pixel":
                # Compute arm position for display
                px, py = args["pixel_x"], args["pixel_y"]
                z_off = args.get("z_offset_mm", 50)
                arm_desc = ""
                if self._last_depth_frame is not None:
                    depth_mm = self.ct.get_depth_at_pixel(self._last_depth_frame, px, py)
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
            elif tool_name == "move_home":
                action_str = "MOVE ARM to HOME position"
            elif tool_name == "gripper_ctrl":
                state = "OPEN" if args["angle"] > 45 else "CLOSE"
                action_str = f"GRIPPER {state} to {args['angle']:.1f} degrees"
            else:
                action_str = f"Execute {tool_name} with args: {json.dumps(args)}"

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

        executor = executors.get(tool_name)
        if executor is None:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}, None

        try:
            print(f"Executing {tool_name}...")
            result = executor(args)
            return result, None
        except Exception as e:
            return {"status": "error", "message": str(e)}, None

    # ── Agent loop ──────────────────────────────────────────────

    def process_task(self, task: str) -> str:
        """Process a task using the vision-driven agent loop."""
        logger.info(f"Processing task: {task}")

        task_prompt = create_task_prompt(task)

        # Build initial messages (text only — LLM will call look() for vision)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Agent iteration {iteration} ---")

            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tools,
                    "tool_choice": "auto",
                }

                if "o1" in self.model.lower() or "o3" in self.model.lower():
                    request_kwargs["extra_body"] = {
                        "reasoning_effort": self.reasoning_effort,
                    }

                response = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise

            if not response.choices:
                logger.warning("No choices in response")
                return "Model returned empty response."

            choice = response.choices[0]
            assistant_message = choice.message

            if assistant_message.content:
                print(f"\nAgent: {assistant_message.content}")

            messages.append(assistant_message.model_dump())

            if not assistant_message.tool_calls:
                return assistant_message.content or "Task completed."

            # Execute tool calls
            movement_executed = False
            pending_images: list[dict] | None = None

            for tool_call in assistant_message.tool_calls:
                result, images = self._execute_tool(tool_call)
                print(f"Result: {result if isinstance(result, str) else json.dumps(result, indent=2)[:200]}")

                # For look/segment, result is a string; for others, it's a dict
                result_str = result if isinstance(result, str) else json.dumps(result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

                if images:
                    pending_images = images

                tool_name = tool_call.function.name
                if tool_name in ["goto_pixel", "move_home", "gripper_ctrl"]:
                    result_dict = result if isinstance(result, dict) else {}
                    if result_dict.get("status") == "success":
                        movement_executed = True

            # Add images from look/segment as a user message
            if pending_images:
                messages.append({
                    "role": "user",
                    "content": pending_images,
                })

            # After movement, capture a fresh view for verification
            if movement_executed:
                print("Capturing post-movement view...")
                time.sleep(0.5)
                text_result, image_content = self._execute_look({})

                messages.append({
                    "role": "user",
                    "content": [
                        *image_content,
                        {
                            "type": "text",
                            "text": (
                                "[Updated view after movement. "
                                "Confirm task completion or continue if more actions needed.]"
                            ),
                        },
                    ],
                })

        return "Maximum iterations reached. Task may be incomplete."

    # ── Interactive REPL ────────────────────────────────────────

    def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        self.start()

        print("\n" + "=" * 60)
        print("Agent V2 - Vision-Driven Interactive Mode (SAM2)")
        print("=" * 60)
        print("Commands:")
        print("  quit       - Exit the agent")
        print("  home       - Move arm to home position")
        print("  pos        - Show current arm position (ground-relative)")
        print("  calibrate  - Run calibration procedure")
        print("  segment    - Interactive segment test (click to segment, no LLM)")
        print("  touch      - Touch debug (click to move arm, then reset)")
        print()
        print("Or enter a task like: 'touch the cup'")
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
                    print(f"Current position: x={x:.1f}, y={y:.1f}, z={z:.1f} (ground-relative)")
                    continue

                if task.lower() == "calibrate":
                    self._run_calibration()
                    continue

                if task.lower() in ["segment", "detect"]:
                    self._run_segment_test()
                    continue

                if task.lower() == "touch":
                    self._run_touch_debug()
                    continue

                print("\nProcessing task...")
                try:
                    response = self.process_task(task)
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

    def _run_segment_test(self) -> None:
        """Interactive segment test: click a point to see SAM2 segmentation.

        Shows the camera image. Click anywhere to run SAM2 point-prompt
        segmentation and see the mask overlay, centroid, and 3D arm coords.

        Press 'r' to refresh the camera frame. Press 'q' to exit.
        """
        window_name = "Segment Test - Click to segment (R=refresh, Q=quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        clicked_pixel: list[tuple[int, int] | None] = [None]

        def mouse_callback(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_pixel[0] = (x, y)

        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "=" * 60)
        print("SEGMENT TEST MODE")
        print("=" * 60)
        print("Click on the image to segment an object at that point.")
        print("Press 'r' to refresh the camera frame.")
        print("Press 'q' to exit.\n")

        # Capture and encode initial frame
        print("Capturing frame and encoding for SAM2...")
        color_image, depth_image, depth_frame = self._capture_aligned_frames()
        encode_time = self.sam2.set_image(color_image)
        print(f"SAM2 image encoded in {encode_time:.1f}s")

        display = color_image.copy()
        cv2.putText(
            display, "Click to segment | R=refresh | Q=quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        try:
            while True:
                cv2.imshow(window_name, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("r"):
                    print("Refreshing frame...")
                    color_image, depth_image, depth_frame = self._capture_aligned_frames()
                    encode_time = self.sam2.set_image(color_image)
                    print(f"SAM2 image encoded in {encode_time:.1f}s")
                    display = color_image.copy()
                    cv2.putText(
                        display, "Click to segment | R=refresh | Q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    clicked_pixel[0] = None
                    continue

                if clicked_pixel[0] is None:
                    continue

                px, py = clicked_pixel[0]
                clicked_pixel[0] = None

                # Run segmentation
                print(f"Segmenting at ({px}, {py})...")
                t0 = time.time()
                mask, score, bbox = self.sam2.segment_point(px, py)
                seg_time = time.time() - t0
                cx, cy = self.sam2.mask_centroid(mask)

                # Get 3D coords
                arm_str = "no calibration"
                depth_mm = self.ct.get_depth_at_pixel(depth_frame, cx, cy)
                if depth_mm is not None:
                    cam_3d = self.ct.deproject_pixel(cx, cy, depth_mm=depth_mm)
                    if cam_3d is not None:
                        arm_3d = self.ct.camera_to_arm(cam_3d)
                        if arm_3d is not None:
                            arm_str = f"arm=({arm_3d[0]:.0f}, {arm_3d[1]:.0f}, {arm_3d[2]:.0f})"

                print(
                    f"  Score: {score:.3f} | Centroid: ({cx},{cy}) | "
                    f"Area: {int(mask.sum())}px | {arm_str} | {seg_time:.2f}s"
                )

                # Draw annotation
                display = self._annotate_segmentation(
                    color_image, mask, bbox, (px, py), (cx, cy)
                )
                cv2.putText(
                    display,
                    f"Score={score:.3f} Centroid=({cx},{cy}) {arm_str}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                cv2.putText(
                    display, "Click to segment | R=refresh | Q=quit",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

        finally:
            cv2.destroyAllWindows()
            print("Segment test ended.")

    def _run_calibration(self) -> None:
        """Run the interactive calibration procedure."""
        from .calibration import CalibrationProcedure

        try:
            proc = CalibrationProcedure(self.camera, self.motion, self.ct)
            R, t, rmse = proc.run(save_path=self._calibration_path)
            # Reload calibration into coordinate transform
            self.ct._R = R
            self.ct._t = t
            print("Calibration loaded into coordinate transform.")
        except Exception as e:
            print(f"Calibration failed: {e}")

    def _run_touch_debug(self) -> None:
        """Interactive touch debug: click a point, arm moves there, then resets.

        Shows side-by-side color + depth (like calibration). Click on
        either image to pick a target. The pixel is deprojected to 3D,
        transformed to arm coordinates, and the arm moves there. After a
        pause, the arm returns home and waits for the next click.

        Press 'q' to exit.
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
                    x = x - color_width[0]  # remap depth-side click to color coords
                clicked_pixel[0] = (x, y)

        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "=" * 60)
        print("TOUCH DEBUG MODE")
        print("=" * 60)
        print("Click on either image to move the arm there.")
        print("The arm will move to the clicked 3D point, pause, then go home.")
        print("Press 'q' in the window to exit.\n")

        # Move home first
        print("Moving arm to home position...")
        self.motion.home()
        time.sleep(1.0)

        try:
            while True:
                # Capture aligned frames
                color_image, depth_image, depth_frame = self._capture_aligned_frames()
                color_width[0] = color_image.shape[1]

                color_display = color_image.copy()
                depth_display = depth_image.copy()

                # Draw instructions
                cv2.putText(
                    color_display,
                    "Click to move arm | Q=quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Show current arm position
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
                    # Draw crosshair on both
                    for img in [color_display, depth_display]:
                        cv2.circle(img, (cx, cy), 8, (0, 0, 255), 2)
                        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
                        cv2.line(img, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)

                # Labels
                cv2.putText(
                    color_display, "COLOR", (10, color_display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                cv2.putText(
                    depth_display, "DEPTH", (10, depth_display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

                combined = np.hstack([color_display, depth_display])
                cv2.imshow(window_name, combined)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    break

                if clicked_pixel[0] is None:
                    continue

                # Process the click
                px, py = clicked_pixel[0]
                clicked_pixel[0] = None

                # Get depth
                depth_mm = self.ct.get_depth_at_pixel(depth_frame, px, py)
                if depth_mm is None:
                    print(f"  No depth at pixel ({px}, {py}) — skipping.")
                    continue

                # Deproject to camera 3D
                cam_3d = self.ct.deproject_pixel(px, py, depth_mm=depth_mm)
                if cam_3d is None:
                    print(f"  Deprojection failed at ({px}, {py}) — skipping.")
                    continue

                # Transform to arm coordinates
                arm_3d = self.ct.camera_to_arm(cam_3d)
                if arm_3d is None:
                    print(f"  Camera→arm transform failed — skipping.")
                    continue

                target_x, target_y, target_z = float(arm_3d[0]), float(arm_3d[1]), float(arm_3d[2])

                print(
                    f"\n  Click: pixel=({px},{py}) depth={depth_mm:.0f}mm\n"
                    f"    cam_3d = [{cam_3d[0]:.1f}, {cam_3d[1]:.1f}, {cam_3d[2]:.1f}]\n"
                    f"    arm_3d = [{target_x:.1f}, {target_y:.1f}, {target_z:.1f}]"
                )

                # Attempt to move
                ok = self.motion.move_to(target_x, target_y, target_z)
                if ok:
                    print(f"    Moved to target. Holding for 2 seconds...")
                    time.sleep(2.0)
                else:
                    print(f"    Move failed — position may be unreachable.")
                    time.sleep(0.5)

                # Return home
                print("    Returning home...")
                self.motion.home()
                time.sleep(1.0)

        finally:
            cv2.destroyAllWindows()
            print("Touch debug ended.")
