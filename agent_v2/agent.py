"""AgentV2: Object-aware robot arm control orchestrator."""

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

from .annotate import annotate_frame
from .calibration import CalibrationProcedure, DEFAULT_CALIBRATION_PATH
from .coordinate_transform import CoordinateTransform
from .models import DetectedObject, SceneState
from .prompts import SYSTEM_PROMPT, create_task_prompt
from .tools import TOOL_DECLARATIONS
from .vision.base import VisionBackend

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
    """Object-aware robot arm control agent.

    Separates perception (vision backend) from reasoning (LLM) from
    execution (motor commands). The LLM references objects by ID
    instead of reasoning about raw pixel coordinates.

    Uses the Motion controller for arm movement (IK-based, ground-relative
    coordinates, torque-based ground probing).
    """

    def __init__(
        self,
        openai_api_key: str,
        helicone_api_key: str,
        vision_backend: VisionBackend,
        model: str = "gpt-4.1-mini",
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

        logger.info(f"Initializing AgentV2 with model={model}, vision={vision_backend.name()}")

        # OpenAI client via Helicone
        self.client = create_openai_client(openai_api_key, helicone_api_key, debug=debug)
        self.model = model

        # Hardware
        self.camera = RealSenseCamera()
        self.motion = Motion(port=arm_port, inverted=True)

        # Vision
        self.vision = vision_backend

        # Coordinate transform
        self.ct = CoordinateTransform(calibration_path=calibration_path)
        self._calibration_path = calibration_path

        # Confirmation
        self.confirmation = ConfirmationHandler(auto_confirm=auto_confirm)

        # Tools
        self.tools = convert_tools_to_openai_format()

        # Current scene state (updated by describe_scene)
        self._current_scene: SceneState | None = None

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

    def _run_perception(self, confidence_threshold: float = 0.3) -> tuple[SceneState, np.ndarray, np.ndarray]:
        """Run the full perception pipeline.

        Returns:
            scene: SceneState with enriched detections
            annotated_image: Color image with bounding boxes drawn
            depth_image: Colorized depth image
        """
        color_image, depth_image, depth_frame = self._capture_aligned_frames()
        h, w = color_image.shape[:2]

        # Detect objects
        detections = self.vision.detect(color_image, confidence_threshold=confidence_threshold)

        # Build scene
        scene = SceneState(objects=detections, image_width=w, image_height=h)

        # Enrich with depth + 3D coordinates
        self.ct.enrich_scene(scene, depth_frame)

        # Annotate frame
        annotated = annotate_frame(color_image, detections)

        self._current_scene = scene
        return scene, annotated, depth_image

    def _encode_image(self, image: np.ndarray, quality: int = 85) -> str:
        """Encode image as base64 JPEG."""
        _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(jpeg.tobytes()).decode("utf-8")

    # ── Tool executors ──────────────────────────────────────────

    def _execute_describe_scene(self, _args: dict[str, Any]) -> tuple[str, list[dict]]:
        """Execute describe_scene tool. Returns (text_result, image_content_blocks)."""
        scene, annotated, depth_image = self._run_perception()

        scene_json = json.dumps(scene.to_dict(), indent=2)
        text_result = f"Detected {len(scene.objects)} objects:\n{scene_json}"

        # Build image content blocks for the LLM response
        annotated_b64 = self._encode_image(annotated)
        depth_b64 = self._encode_image(depth_image)

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
                "text": "[Annotated camera image above with object bounding boxes. Depth image below.]",
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

    def _execute_goto(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute goto tool — move arm to a detected object."""
        object_id = args["object_id"]
        z_offset_mm = args.get("z_offset_mm", 50)

        if self._current_scene is None:
            return {
                "status": "error",
                "message": "No scene available. Call describe_scene() first.",
            }

        obj = self._current_scene.get_object(object_id)
        if obj is None:
            available = [o.object_id for o in self._current_scene.objects]
            return {
                "status": "error",
                "message": f"Object ID {object_id} not found. Available IDs: {available}",
            }

        if obj.position_arm is None:
            return {
                "status": "error",
                "message": (
                    f"Object {object_id} ({obj.label}) has no arm coordinates. "
                    "Depth may be missing or calibration not loaded."
                ),
            }

        target_x = obj.position_arm.x
        target_y = obj.position_arm.y
        target_z = obj.position_arm.z + z_offset_mm

        ok = self.motion.move_to(target_x, target_y, target_z)
        if ok:
            return {
                "status": "success",
                "message": f"Moved to {obj.label} (id={object_id}) at x={target_x:.0f}, y={target_y:.0f}, z={target_z:.0f}",
            }
        else:
            return {
                "status": "error",
                "message": f"move_to failed — position ({target_x:.0f}, {target_y:.0f}, {target_z:.0f}) may be unreachable",
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

        # describe_scene doesn't need confirmation
        if tool_name == "describe_scene":
            print("Running perception pipeline...")
            text_result, images = self._execute_describe_scene(args)
            return text_result, images

        # Confirmation for movement tools
        needs_confirm = tool_name in ["goto", "move_home", "gripper_ctrl"]
        if needs_confirm and not self.confirmation.auto_confirm:
            # Build human-readable description
            if tool_name == "goto":
                obj_desc = f"object_id={args['object_id']}"
                if self._current_scene:
                    obj = self._current_scene.get_object(args["object_id"])
                    if obj:
                        obj_desc = f"{obj.label} (id={args['object_id']})"
                        if obj.position_arm:
                            obj_desc += (
                                f" at x={obj.position_arm.x:.0f}, "
                                f"y={obj.position_arm.y:.0f}, "
                                f"z={obj.position_arm.z:.0f}"
                            )
                z_off = args.get("z_offset_mm", 50)
                action_str = f"GOTO {obj_desc} (z_offset={z_off}mm)"
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
            "goto": self._execute_goto,
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
        """Process a task using the object-aware agent loop."""
        logger.info(f"Processing task: {task}")

        task_prompt = create_task_prompt(task)

        # Build initial messages (text only — LLM will call describe_scene for vision)
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

                # For describe_scene, result is a string; for others, it's a dict
                result_str = result if isinstance(result, str) else json.dumps(result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

                if images:
                    pending_images = images

                tool_name = tool_call.function.name
                if tool_name in ["goto", "move_home", "gripper_ctrl"]:
                    result_dict = result if isinstance(result, dict) else {}
                    if result_dict.get("status") == "success":
                        movement_executed = True

            # Add images from describe_scene as a user message
            if pending_images:
                messages.append({
                    "role": "user",
                    "content": pending_images,
                })

            # After movement, capture a fresh annotated frame for verification
            if movement_executed:
                print("Capturing post-movement view...")
                time.sleep(0.5)
                scene, annotated, depth_image = self._run_perception()
                scene_json = json.dumps(scene.to_dict(), indent=2)

                annotated_b64 = self._encode_image(annotated)
                depth_b64 = self._encode_image(depth_image)

                messages.append({
                    "role": "user",
                    "content": [
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
                                f"[Updated view after movement. Current detections:\n{scene_json}\n"
                                "Confirm task completion or continue if more actions needed.]"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{depth_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                })

        return "Maximum iterations reached. Task may be incomplete."

    # ── Interactive REPL ────────────────────────────────────────

    def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        self.start()

        print("\n" + "=" * 60)
        print("Agent V2 - Object-Aware Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  quit       - Exit the agent")
        print("  home       - Move arm to home position")
        print("  pos        - Show current arm position (ground-relative)")
        print("  calibrate  - Run calibration procedure")
        print("  detect     - Run perception only (no LLM)")
        print("  touch      - Touch debug (click to move arm, then reset)")
        print()
        print("Or enter a task like: 'go to the cup'")
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

                if task.lower() == "detect":
                    self._run_detect_only()
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

    def _run_calibration(self) -> None:
        """Run the interactive calibration procedure."""
        try:
            proc = CalibrationProcedure(self.camera, self.motion, self.ct)
            R, t, rmse = proc.run(save_path=self._calibration_path)
            # Reload calibration into coordinate transform
            self.ct._R = R
            self.ct._t = t
            print("Calibration loaded into coordinate transform.")
        except Exception as e:
            print(f"Calibration failed: {e}")

    def _run_detect_only(self) -> None:
        """Run perception pipeline and print results (no LLM)."""
        print("Running perception pipeline...")
        scene, annotated, _depth = self._run_perception()
        print(json.dumps(scene.to_dict(), indent=2))

        # Show annotated image in OpenCV window
        cv2.imshow("Detections", annotated)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
