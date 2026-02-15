"""AgentV3: Multi-provider robot arm control orchestrator with dynamic actions."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

from motion_controller.motion import Motion

from .calibration import DEFAULT_CALIBRATION_PATH
from .camera import RealSenseCamera
from .coordinate_transform import CoordinateTransform
from .llm import ConfirmationHandler, LLMProvider
from .prompts import create_system_prompt, create_task_prompt
from .tools import get_tool_declarations

logger = logging.getLogger("agent")

# Gripper defaults (for direct SDK calls — Motion doesn't wrap gripper)
GRIPPER_SPEED = 100
GRIPPER_ACC = 50

# Tools that need user confirmation before execution
CONFIRM_TOOLS = {
    "goto_pixel",
    "move_home",
    "gripper_ctrl",
    "move_to_xyz",
    "move_relative",
    "press_down",
    "force_move",
    "execute_trajectory",
}


class AgentV3:
    """Multi-provider robot arm control agent with dynamic action capabilities.

    Supports OpenAI GPT or Gemini ER for reasoning, SAM2 and/or Gemini ER
    for vision, and force-controlled / trajectory-based dynamic actions.
    """

    def __init__(
        self,
        provider: LLMProvider,
        arm_port: str | None = None,
        auto_confirm: bool = False,
        debug: bool = False,
        calibration_path: Path = DEFAULT_CALIBRATION_PATH,
        sam2_backend: Any = None,
        gemini_vision: Any = None,
    ):
        self.debug = debug
        self.provider = provider

        if debug:
            logging.getLogger("agent_v3").setLevel(logging.DEBUG)

        logger.info("Initializing AgentV3")

        # Hardware
        self.camera = RealSenseCamera()
        self.motion = Motion(port=arm_port, inverted=True)

        # Coordinate transform
        self.ct = CoordinateTransform(calibration_path=calibration_path)
        self._calibration_path = calibration_path

        # Confirmation
        self.confirmation = ConfirmationHandler(auto_confirm=auto_confirm)

        # Vision backends (both optional, can work in tandem)
        self.sam2 = sam2_backend
        self.gemini_vision = gemini_vision

        # Tool declarations (filtered by available vision)
        self.tool_declarations = get_tool_declarations(
            has_sam2=self.sam2 is not None,
            has_gemini_vision=self.gemini_vision is not None,
        )

        # Frame cache (auto-captured each turn, used by segment/detect/goto_pixel)
        self._last_color_image: np.ndarray | None = None
        self._last_depth_frame: rs.depth_frame | None = None
        self._last_depth_image: np.ndarray | None = None
        self._sam2_needs_update: bool = True

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
        """Capture aligned color + depth frames."""
        if not self.camera._started:
            self.camera.start()

        frames = self.camera.pipeline.wait_for_frames()
        aligned = self.ct.get_aligned_frames(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")

        color_image = np.asanyarray(color_frame.get_data())
        colorized = self.camera.colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized.get_data())
        depth_image = self.camera._draw_depth_scale(
            depth_image.copy(), self.camera.min_depth_m, self.camera.max_depth_m
        )

        return color_image, depth_image, depth_frame

    def _encode_image(self, image: np.ndarray, quality: int = 85) -> bytes:
        """Encode a BGR numpy image as JPEG bytes."""
        _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

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

    # ── Auto-inject frame & history rewriting ─────────────────

    def _capture_and_inject_frame(self, messages: list) -> None:
        """Capture a fresh camera frame and inject it as a user message.

        Caches color/depth for detect/segment/goto_pixel. Sets _sam2_needs_update.
        SAM2 encoding is deferred to _execute_segment() (lazy).
        """
        color_image, depth_image, depth_frame = self._capture_aligned_frames()

        self._last_color_image = color_image
        self._last_depth_frame = depth_frame
        self._last_depth_image = depth_image
        self._sam2_needs_update = True

        color_with_grid = self._draw_coordinate_grid(color_image)

        encoded_parts = [
            self.provider.encode_image(
                self._encode_image(color_with_grid), "image/jpeg"
            ),
            self.provider.encode_text(
                "[Camera image with coordinate grid (labels every 80px). Depth image below.]"
            ),
            self.provider.encode_image(self._encode_image(depth_image), "image/jpeg"),
        ]
        self.provider.append_images(messages, encoded_parts)

    def _rewrite_image_history(self, messages: list) -> None:
        """Replace all existing image messages with LLM-generated text summaries.

        For each user message containing images, finds the next assistant/model
        message and uses its text as the summary.
        """
        image_indices = self.provider.find_image_message_indices(messages)

        for idx in image_indices:
            # Find the next assistant message after this image
            summary = None
            for j in range(idx + 1, len(messages)):
                text = self.provider.extract_assistant_text(messages, j)
                if text:
                    summary = text[:500] + "..." if len(text) > 500 else text
                    break

            if summary is None:
                summary = "Frame was captured but no observation was recorded."

            self.provider.replace_images_with_text(messages, idx, summary)

    # ── Tool executors ──────────────────────────────────────────

    def _execute_detect(self, args: dict[str, Any]) -> tuple[str, list[dict]]:
        """Execute detect tool: Gemini ER segmentation with natural language query."""
        query = args["query"]

        if self._last_color_image is None or self._last_depth_frame is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No image available (frame not yet captured).",
                }
            ), []

        if self.gemini_vision is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Gemini vision not available.",
                }
            ), []

        print(f"Detecting: {query!r}...")
        segments = self.gemini_vision.segment(self._last_color_image, query)

        if not segments:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"No objects matching '{query}' detected.",
                }
            ), []

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

        image_content = [
            {
                "type": "image",
                "data": self._encode_image(annotated),
                "mime_type": "image/jpeg",
            },
            {
                "type": "text",
                "text": (
                    f"[Detection result for '{query}'. Label: {label}, "
                    f"Centroid: ({cx},{cy}), Mask area: {mask_area}px]"
                ),
            },
        ]

        return text_result, image_content

    def _execute_segment(self, args: dict[str, Any]) -> tuple[str, list[dict]]:
        """Execute segment tool: SAM2 point-prompt segmentation."""
        pixel_x = args["pixel_x"]
        pixel_y = args["pixel_y"]

        if self._last_color_image is None or self._last_depth_frame is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No image available (frame not yet captured).",
                }
            ), []

        if self.sam2 is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "SAM2 not available.",
                }
            ), []

        # Lazy SAM2 encoding — only when segment is actually called
        if self._sam2_needs_update:
            print("Encoding image for SAM2...")
            encode_time = self.sam2.set_image(self._last_color_image)
            logger.info(f"SAM2 image encoded in {encode_time:.2f}s")
            self._sam2_needs_update = False

        print(f"Running SAM2 segmentation at ({pixel_x}, {pixel_y})...")
        t0 = time.time()
        mask, score, bbox = self.sam2.segment_point(pixel_x, pixel_y)
        seg_time = time.time() - t0
        logger.info(f"SAM2 segmentation in {seg_time:.2f}s, score={score:.3f}")

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

        image_content = [
            {
                "type": "image",
                "data": self._encode_image(annotated),
                "mime_type": "image/jpeg",
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

        if mask is not None and mask.any():
            green_overlay = annotated.copy()
            green_overlay[mask] = [0, 200, 0]
            cv2.addWeighted(green_overlay, 0.4, annotated, 0.6, 0, annotated)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        mx, my = centroid
        cv2.circle(annotated, (mx, my), 6, (255, 0, 0), -1)
        cv2.circle(annotated, (mx, my), 6, (255, 255, 255), 1)

        annotated = self._draw_coordinate_grid(annotated)
        return annotated

    def _annotate_segmentation(
        self,
        color_image: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        click: tuple[int, int],
    ) -> np.ndarray:
        """Draw segmentation annotation on the image (ported from V2)."""
        annotated = color_image.copy()

        green_overlay = annotated.copy()
        green_overlay[mask] = [0, 200, 0]
        cv2.addWeighted(green_overlay, 0.4, annotated, 0.6, 0, annotated)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cx, cy = click
        cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        annotated = self._draw_coordinate_grid(annotated)
        return annotated

    # ── Movement tool executors ────────────────────────────────

    def _execute_goto_pixel(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute goto_pixel: pixel -> depth -> deproject -> affine -> move."""
        pixel_x = args["pixel_x"]
        pixel_y = args["pixel_y"]
        z_offset_mm = args.get("z_offset_mm", 50)

        if self._last_depth_frame is None:
            return {
                "status": "error",
                "message": "No depth frame available (frame not yet captured).",
            }

        depth_mm = self.ct.get_depth_at_pixel(self._last_depth_frame, pixel_x, pixel_y)
        if depth_mm is None:
            return {
                "status": "error",
                "message": f"No depth at pixel ({pixel_x}, {pixel_y}).",
            }

        cam_3d = self.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
        if cam_3d is None:
            return {
                "status": "error",
                "message": f"Deprojection failed at ({pixel_x}, {pixel_y}).",
            }

        arm_3d = self.ct.camera_to_arm(cam_3d)
        if arm_3d is None:
            return {"status": "error", "message": "Camera-to-arm transform failed."}

        target_x = float(arm_3d[0])
        target_y = float(arm_3d[1])
        target_z = float(arm_3d[2]) + z_offset_mm

        ok = self.motion.move_to(target_x, target_y, target_z)
        if ok:
            return {
                "status": "success",
                "message": f"Moved to pixel ({pixel_x},{pixel_y}) -> arm x={target_x:.0f}, y={target_y:.0f}, z={target_z:.0f}",
                "pixel": {"x": pixel_x, "y": pixel_y},
                "arm_position": {
                    "x": round(target_x, 1),
                    "y": round(target_y, 1),
                    "z": round(target_z, 1),
                },
            }
        return {
            "status": "error",
            "message": f"Position ({target_x:.0f}, {target_y:.0f}, {target_z:.0f}) unreachable",
        }

    def _execute_move_to_xyz(self, args: dict[str, Any]) -> dict[str, Any]:
        x, y, z = args["x"], args["y"], args["z"]
        ok = self.motion.move_to(x, y, z)
        if ok:
            final = self.motion.get_pose()
            return {
                "status": "success",
                "message": f"Moved to ({x:.0f}, {y:.0f}, {z:.0f})",
                "position": {
                    "x": round(final[0], 1),
                    "y": round(final[1], 1),
                    "z": round(final[2], 1),
                },
            }
        return {
            "status": "error",
            "message": f"Position ({x:.0f}, {y:.0f}, {z:.0f}) unreachable",
        }

    def _execute_move_relative(self, args: dict[str, Any]) -> dict[str, Any]:
        dx, dy, dz = args["dx"], args["dy"], args["dz"]
        ok = self.motion.move_relative(dx, dy, dz)
        if ok:
            final = self.motion.get_pose()
            return {
                "status": "success",
                "message": f"Moved relative ({dx:.0f}, {dy:.0f}, {dz:.0f})",
                "position": {
                    "x": round(final[0], 1),
                    "y": round(final[1], 1),
                    "z": round(final[2], 1),
                },
            }
        return {
            "status": "error",
            "message": f"Relative move ({dx:.0f}, {dy:.0f}, {dz:.0f}) failed",
        }

    def _execute_press_down(self, args: dict[str, Any]) -> dict[str, Any]:
        max_force = args.get("max_force", 50)
        max_dist = args.get("max_distance_mm", 50.0)
        result = self.motion.force_move(
            direction=(0, 0, -1),
            max_force=max_force,
            max_distance_mm=max_dist,
        )
        return {
            "status": "success",
            "contact": result["contact"],
            "distance_mm": result["distance_mm"],
            "max_torque_delta": result["max_torque_delta"],
            "position": {
                "x": round(result["final_position"][0], 1),
                "y": round(result["final_position"][1], 1),
                "z": round(result["final_position"][2], 1),
            },
            "message": (
                f"{'Contact detected' if result['contact'] else 'No contact'} "
                f"after {result['distance_mm']:.1f}mm descent"
            ),
        }

    def _execute_force_move(self, args: dict[str, Any]) -> dict[str, Any]:
        dx, dy, dz = args["dx"], args["dy"], args["dz"]
        max_force = args.get("max_force", 50)
        max_dist = args.get("max_distance_mm", 100.0)
        result = self.motion.force_move(
            direction=(dx, dy, dz),
            max_force=max_force,
            max_distance_mm=max_dist,
        )
        return {
            "status": "success",
            "contact": result["contact"],
            "distance_mm": result["distance_mm"],
            "max_torque_delta": result["max_torque_delta"],
            "position": {
                "x": round(result["final_position"][0], 1),
                "y": round(result["final_position"][1], 1),
                "z": round(result["final_position"][2], 1),
            },
            "message": (
                f"{'Contact detected' if result['contact'] else 'Distance exhausted'} "
                f"after {result['distance_mm']:.1f}mm travel "
                f"(peak torque delta: {result['max_torque_delta']})"
            ),
        }

    def _execute_trajectory(self, args: dict[str, Any]) -> dict[str, Any]:
        waypoints_raw = args["waypoints"]
        force_threshold = args.get("force_threshold")

        waypoints = [(w[0], w[1], w[2]) for w in waypoints_raw]

        if len(waypoints) < 2:
            return {"status": "error", "message": "At least 2 waypoints required"}

        result = self.motion.follow_trajectory(
            waypoints, force_threshold=force_threshold
        )
        return {
            "status": "success" if result["completed"] else "partial",
            "completed": result["completed"],
            "waypoints_executed": result["waypoints_executed"],
            "total_waypoints": len(waypoints),
            "contact": result.get("contact", False),
            "position": {
                "x": round(result["final_position"][0], 1),
                "y": round(result["final_position"][1], 1),
                "z": round(result["final_position"][2], 1),
            },
            "message": (
                f"Executed {result['waypoints_executed']}/{len(waypoints)} waypoints"
                + (" (stopped: contact)" if result.get("contact") else "")
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

    # ── Tool dispatch ──────────────────────────────────────────

    def _execute_tool(
        self, name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any] | str, list[dict] | None]:
        """Execute a tool call. Returns (result, optional_image_content_blocks)."""
        # Vision tools return images, no confirmation needed
        if name == "detect":
            print(f"Detecting: {args.get('query', '?')}...")
            return self._execute_detect(args)

        if name == "segment":
            print(f"Segmenting at ({args.get('pixel_x')}, {args.get('pixel_y')})...")
            return self._execute_segment(args)

        # Confirmation for movement tools
        if name in CONFIRM_TOOLS and not self.confirmation.auto_confirm:
            action_str = self._format_action(name, args)

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
            "move_to_xyz": self._execute_move_to_xyz,
            "move_relative": self._execute_move_relative,
            "press_down": self._execute_press_down,
            "force_move": self._execute_force_move,
            "execute_trajectory": self._execute_trajectory,
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

    def _format_action(self, name: str, args: dict[str, Any]) -> str:
        """Format a pending action for display."""
        if name == "goto_pixel":
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
                            arm_desc = f" -> arm x={arm_3d[0]:.0f}, y={arm_3d[1]:.0f}, z={float(arm_3d[2]) + z_off:.0f}"
            return f"GOTO pixel ({px},{py}) z_offset={z_off}mm{arm_desc}"
        elif name == "move_to_xyz":
            return f"MOVE ARM to ({args['x']:.0f}, {args['y']:.0f}, {args['z']:.0f})"
        elif name == "move_relative":
            return f"MOVE ARM relative ({args['dx']:.0f}, {args['dy']:.0f}, {args['dz']:.0f})"
        elif name == "press_down":
            return f"PRESS DOWN max_force={args.get('max_force', 50)} max_dist={args.get('max_distance_mm', 50)}mm"
        elif name == "force_move":
            return f"FORCE MOVE dir=({args['dx']:.0f},{args['dy']:.0f},{args['dz']:.0f}) max_force={args.get('max_force', 50)}"
        elif name == "execute_trajectory":
            n = len(args.get("waypoints", []))
            return f"EXECUTE TRAJECTORY with {n} waypoints"
        elif name == "move_home":
            return "MOVE ARM to HOME position"
        elif name == "gripper_ctrl":
            state = "OPEN" if args["angle"] > 45 else "CLOSE"
            return f"GRIPPER {state} to {args['angle']:.1f} degrees"
        return f"Execute {name} with args: {json.dumps(args)}"

    # ── Agent loop ──────────────────────────────────────────────

    def process_task(self, task: str) -> str:
        """Process a task using the agent loop (provider-agnostic).

        Each iteration: rewrite old images → inject fresh frame → LLM → execute tools.
        """
        logger.info(f"Processing task: {task}")

        system_prompt = create_system_prompt(
            has_sam2=self.sam2 is not None,
            has_gemini_vision=self.gemini_vision is not None,
        )
        task_prompt = create_task_prompt(task)

        messages = self.provider.create_messages(system_prompt, task_prompt)

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Agent iteration {iteration} ---")

            # Step 1: Replace old image messages with text summaries
            self._rewrite_image_history(messages)

            # Step 2: Auto-inject fresh camera frame
            print("Capturing camera frame...")
            self._capture_and_inject_frame(messages)

            # Step 3: Call LLM
            try:
                response = self.provider.chat(messages, self.tool_declarations)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise

            if response.text:
                print(f"\nAgent: {response.text}")

            # Step 4: Add model response to conversation
            self.provider.append_assistant_response(messages, response)

            # Step 5: If no tool calls, task is done
            if not response.tool_calls:
                return response.text or "Task completed."

            # Step 6: Execute tool calls
            pending_image_blocks: list[dict] = []

            for tc in response.tool_calls:
                result, images = self._execute_tool(tc.name, tc.args)
                result_str = result if isinstance(result, str) else json.dumps(result)
                print(f"Result: {result_str[:200]}")

                self.provider.append_tool_result(messages, tc, result_str)

                if images:
                    pending_image_blocks.extend(images)

            # Step 7: Add images from detect/segment as a user message
            # (will be rewritten to text summaries on next iteration)
            if pending_image_blocks:
                encoded_parts = []
                for block in pending_image_blocks:
                    if block["type"] == "image":
                        encoded_parts.append(
                            self.provider.encode_image(
                                block["data"], block["mime_type"]
                            )
                        )
                    elif block["type"] == "text":
                        encoded_parts.append(self.provider.encode_text(block["text"]))
                self.provider.append_images(messages, encoded_parts)

        return "Maximum iterations reached. Task may be incomplete."

    # ── Interactive REPL ────────────────────────────────────────

    def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        self.start()

        provider_name = type(self.provider).__name__
        vision_parts = []
        if self.sam2 is not None:
            vision_parts.append("SAM2")
        if self.gemini_vision is not None:
            vision_parts.append("Gemini ER")
        vision_str = " + ".join(vision_parts) if vision_parts else "none"

        print("\n" + "=" * 60)
        print("Agent V3 - Interactive Mode")
        print(f"  LLM: {provider_name} | Vision: {vision_str}")
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
        from .calibration import CalibrationProcedure

        try:
            proc = CalibrationProcedure(self.camera, self.motion, self.ct)
            M, rmse = proc.run(save_path=self._calibration_path)
            self.ct._M = M
            print("Calibration loaded into coordinate transform.")
        except Exception as e:
            print(f"Calibration failed: {e}")

    def _run_touch_debug(self) -> None:
        """Interactive touch debug: click a point, arm moves there, then resets."""
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
