"""WebAgentV3: wraps agent_v3 components for web UI with EventBus integration."""

import asyncio
import base64
import json
import logging
import time
from typing import Any

import cv2
import numpy as np

from .events import EventBus
from .hardware import HardwareManager

logger = logging.getLogger("server.agent_wrapper")

GRIPPER_SPEED = 100
GRIPPER_ACC = 50

# Tools that need user confirmation before execution
CONFIRM_TOOLS = {
    "goto_pixel", "move_home", "gripper_ctrl",
    "move_to_xyz", "move_relative", "press_down",
    "force_move", "execute_trajectory",
}

MOVEMENT_TOOLS = CONFIRM_TOOLS


class WebAgentV3:
    """Wraps agent_v3's LLM providers and tools with shared HardwareManager."""

    def __init__(
        self,
        hw: HardwareManager,
        bus: EventBus,
        provider: Any,  # agent_v3.llm.LLMProvider
    ):
        self._hw = hw
        self._bus = bus
        self.provider = provider
        self._state = "idle"  # idle | running | awaiting_confirm | error
        self._current_task: str | None = None
        self._cancel_requested = False
        self._pending_confirmation: asyncio.Future | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # Tool declarations (filtered by available vision)
        from agent_v3.tools import get_tool_declarations

        self.tool_declarations = get_tool_declarations(
            has_sam2=hw.sam2 is not None,
            has_gemini_vision=hw.gemini_vision is not None,
        )

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_task(self) -> str | None:
        return self._current_task

    # ── Task lifecycle (same interface as WebAgentV2) ──────────

    async def submit_task(self, task: str, auto_confirm: bool = False) -> str:
        """Submit a task for async execution. Returns task_id."""
        if self._state == "running":
            raise RuntimeError("Agent is already running a task")

        self._state = "running"
        self._current_task = task
        self._cancel_requested = False
        self._event_loop = asyncio.get_event_loop()

        task_id = f"task_{int(time.time() * 1000)}"

        await self._bus.publish(
            "agent.state_changed", {"state": "running", "task": task}
        )

        asyncio.create_task(
            self._run_in_background(task, auto_confirm, task_id)
        )
        return task_id

    async def _run_in_background(
        self, task: str, auto_confirm: bool, task_id: str
    ):
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, self._execute_task_sync, task, auto_confirm
            )
            await self._bus.publish(
                "agent.task_complete",
                {"result": result, "task_id": task_id},
            )
        except Exception as e:
            logger.error(f"Agent task error: {e}", exc_info=True)
            await self._bus.publish(
                "error", {"code": "agent_error", "message": str(e)}
            )
        finally:
            self._state = "idle"
            self._current_task = None
            await self._bus.publish(
                "agent.state_changed", {"state": "idle", "task": None}
            )

    async def cancel(self):
        self._cancel_requested = True
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(False)

    async def confirm_action(self, approved: bool):
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(approved)

    # ── Helpers ────────────────────────────────────────────────

    def _publish(self, event_type: str, payload: dict):
        """Thread-safe publish from sync context."""
        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._bus.publish(event_type, payload), self._event_loop
            )

    def _encode_image_bytes(self, image: np.ndarray, quality: int = 85) -> bytes:
        """Encode BGR image as JPEG bytes."""
        _, jpeg = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return jpeg.tobytes()

    def _encode_image_b64(self, image: np.ndarray, quality: int = 85) -> str:
        """Encode BGR image as base64 JPEG string."""
        return base64.b64encode(self._encode_image_bytes(image, quality)).decode("utf-8")

    def _wait_for_confirmation(self, action_str: str) -> bool:
        """Block until frontend confirms/rejects. Returns True if approved."""
        self._state = "awaiting_confirm"
        self._publish("agent.state_changed", {"state": "awaiting_confirm", "action": action_str})

        future = asyncio.run_coroutine_threadsafe(
            self._create_confirmation_future(), self._event_loop
        )
        result = future.result(timeout=300)  # 5 min timeout
        self._state = "running"
        self._publish("agent.state_changed", {"state": "running", "task": self._current_task})
        return result

    async def _create_confirmation_future(self) -> bool:
        loop = asyncio.get_event_loop()
        self._pending_confirmation = loop.create_future()
        return await self._pending_confirmation

    # ── Frame capture ──────────────────────────────────────────

    def _capture_aligned_frames(self) -> tuple[np.ndarray, np.ndarray, Any]:
        """Capture aligned color + depth frames."""
        hw = self._hw
        frames = hw.camera.pipeline.wait_for_frames()
        aligned = hw.ct.get_aligned_frames(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")

        color_image = np.asanyarray(color_frame.get_data())
        colorized = hw.camera.colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized.get_data())
        depth_image = hw.camera._draw_depth_scale(
            depth_image.copy(), hw.camera.min_depth_m, hw.camera.max_depth_m
        )

        return color_image, depth_image, depth_frame

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

    # ── Annotation helpers ─────────────────────────────────────

    def _annotate_detection(
        self,
        color_image: np.ndarray,
        mask: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        centroid: tuple[int, int],
        label: str,
    ) -> np.ndarray:
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
            annotated, label, (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
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

    # ── Tool executors ─────────────────────────────────────────

    def _execute_look(self, args: dict) -> tuple[str, list[dict], list[str]]:
        """Capture frame + encode. Returns (text, image_blocks, web_images)."""
        color_image, depth_image, depth_frame = self._capture_aligned_frames()

        # Cache for segment/detect/goto
        self._hw.vision_color = color_image
        self._hw.vision_depth_frame = depth_frame

        # Encode for SAM2 if available
        if self._hw.sam2 is not None:
            encode_time = self._hw.sam2.set_image(color_image)
            text_result = (
                f"Camera frame captured (640x480). "
                f"SAM2 image encoded in {encode_time:.1f}s. "
                f"Ready for segment() and detect() queries."
            )
        else:
            text_result = "Camera frame captured (640x480). Ready for detect() queries."

        # Draw grid on copy, encode
        color_with_grid = self._draw_coordinate_grid(color_image)
        color_jpeg = self._encode_image_bytes(color_with_grid)
        depth_jpeg = self._encode_image_bytes(depth_image)

        # Provider-agnostic image blocks
        image_blocks = [
            {"type": "image", "data": color_jpeg, "mime_type": "image/jpeg"},
            {"type": "text", "text": "[Camera image with coordinate grid. Depth image below.]"},
            {"type": "image", "data": depth_jpeg, "mime_type": "image/jpeg"},
        ]

        # Web UI b64 images
        color_b64 = base64.b64encode(color_jpeg).decode("utf-8")
        depth_b64 = base64.b64encode(depth_jpeg).decode("utf-8")
        web_images = [
            f"data:image/jpeg;base64,{color_b64}",
            f"data:image/jpeg;base64,{depth_b64}",
        ]

        return text_result, image_blocks, web_images

    def _execute_detect(self, args: dict) -> tuple[str, list[dict], list[str]]:
        """Execute detect tool: Gemini ER segmentation with natural language query."""
        query = args["query"]

        if self._hw.vision_color is None or self._hw.vision_depth_frame is None:
            return json.dumps({"status": "error", "message": "No image available. Call look() first."}), [], []

        if self._hw.gemini_vision is None:
            return json.dumps({"status": "error", "message": "Gemini vision not available."}), [], []

        segments = self._hw.gemini_vision.segment(self._hw.vision_color, query)

        if not segments:
            return json.dumps({"status": "error", "message": f"No objects matching '{query}' detected."}), [], []

        seg = segments[0]
        bbox_px = seg.get("box_2d_px", (0, 0, 0, 0))
        label = seg.get("label", query)
        centroid = seg.get("centroid", ((bbox_px[0] + bbox_px[2]) // 2, (bbox_px[1] + bbox_px[3]) // 2))
        cx, cy = centroid

        # Compute 3D arm coordinates
        arm_coords = None
        depth_mm = self._hw.ct.get_depth_at_pixel(self._hw.vision_depth_frame, cx, cy)
        if depth_mm is not None:
            cam_3d = self._hw.ct.deproject_pixel(cx, cy, depth_mm=depth_mm)
            if cam_3d is not None:
                arm_3d = self._hw.ct.camera_to_arm(cam_3d)
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
            "bbox": {"x1": bbox_px[0], "y1": bbox_px[1], "x2": bbox_px[2], "y2": bbox_px[3]},
            "mask_area_px": mask_area,
            "depth_mm": round(depth_mm, 1) if depth_mm else None,
            "arm_coordinates": arm_coords,
        }

        text_result = json.dumps(result)

        annotated = self._annotate_detection(
            self._hw.vision_color, mask, bbox_px, (cx, cy), label
        )
        annotated_jpeg = self._encode_image_bytes(annotated)

        image_blocks = [
            {"type": "image", "data": annotated_jpeg, "mime_type": "image/jpeg"},
            {"type": "text", "text": (
                f"[Detection result for '{query}'. Label: {label}, "
                f"Centroid: ({cx},{cy}), Mask area: {mask_area}px]"
            )},
        ]

        annotated_b64 = base64.b64encode(annotated_jpeg).decode("utf-8")
        web_images = [f"data:image/jpeg;base64,{annotated_b64}"]

        return text_result, image_blocks, web_images

    def _execute_segment(self, args: dict) -> tuple[str, list[dict], list[str]]:
        """Execute segment tool: SAM2 point-prompt segmentation."""
        pixel_x = args["pixel_x"]
        pixel_y = args["pixel_y"]

        if self._hw.vision_color is None or self._hw.vision_depth_frame is None:
            return json.dumps({"status": "error", "message": "No image available. Call look() first."}), [], []

        if self._hw.sam2 is None:
            return json.dumps({"status": "error", "message": "SAM2 not available."}), [], []

        t0 = time.time()
        mask, score, bbox = self._hw.sam2.segment_point(pixel_x, pixel_y)
        seg_time = time.time() - t0

        # Compute 3D arm coordinates
        arm_coords = None
        depth_mm = self._hw.ct.get_depth_at_pixel(self._hw.vision_depth_frame, pixel_x, pixel_y)
        if depth_mm is not None:
            cam_3d = self._hw.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
            if cam_3d is not None:
                arm_3d = self._hw.ct.camera_to_arm(cam_3d)
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
            "bbox": {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])},
            "mask_area_px": int(mask.sum()),
            "depth_mm": round(depth_mm, 1) if depth_mm else None,
            "arm_coordinates": arm_coords,
            "segment_time_s": round(seg_time, 2),
        }

        text_result = json.dumps(result)

        annotated = self._annotate_segmentation(
            self._hw.vision_color, mask, bbox, (pixel_x, pixel_y)
        )
        annotated_jpeg = self._encode_image_bytes(annotated)

        image_blocks = [
            {"type": "image", "data": annotated_jpeg, "mime_type": "image/jpeg"},
            {"type": "text", "text": (
                f"[Segmentation mask overlay. Click target: ({pixel_x},{pixel_y}), "
                f"Score: {score:.3f}]"
            )},
        ]

        annotated_b64 = base64.b64encode(annotated_jpeg).decode("utf-8")
        web_images = [f"data:image/jpeg;base64,{annotated_b64}"]

        return text_result, image_blocks, web_images

    def _execute_goto_pixel(self, args: dict) -> dict:
        hw = self._hw
        pixel_x = args["pixel_x"]
        pixel_y = args["pixel_y"]
        z_offset_mm = args.get("z_offset_mm", 50)

        if hw.vision_depth_frame is None:
            return {"status": "error", "message": "No depth frame. Call look() first."}

        depth_mm = hw.ct.get_depth_at_pixel(hw.vision_depth_frame, pixel_x, pixel_y)
        if depth_mm is None:
            return {"status": "error", "message": f"No depth at pixel ({pixel_x}, {pixel_y})"}

        cam_3d = hw.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
        if cam_3d is None:
            return {"status": "error", "message": "Deprojection failed"}

        arm_3d = hw.ct.camera_to_arm(cam_3d)
        if arm_3d is None:
            return {"status": "error", "message": "Camera-to-arm transform failed"}

        tx, ty, tz = float(arm_3d[0]), float(arm_3d[1]), float(arm_3d[2]) + z_offset_mm

        self._publish("arm.moving", {"target_x": tx, "target_y": ty, "target_z": tz, "status": "started"})
        ok = hw.motion.move_to(tx, ty, tz)
        self._publish("arm.moving", {"target_x": tx, "target_y": ty, "target_z": tz, "status": "reached" if ok else "failed"})

        if ok:
            return {
                "status": "success",
                "message": f"Moved to pixel ({pixel_x},{pixel_y}) -> arm ({tx:.0f},{ty:.0f},{tz:.0f})",
                "pixel": {"x": pixel_x, "y": pixel_y},
                "arm_position": {"x": round(tx, 1), "y": round(ty, 1), "z": round(tz, 1)},
            }
        return {"status": "error", "message": f"Position ({tx:.0f},{ty:.0f},{tz:.0f}) unreachable"}

    def _execute_move_to_xyz(self, args: dict) -> dict:
        x, y, z = args["x"], args["y"], args["z"]
        ok = self._hw.motion.move_to(x, y, z)
        if ok:
            final = self._hw.motion.get_pose()
            return {
                "status": "success",
                "message": f"Moved to ({x:.0f}, {y:.0f}, {z:.0f})",
                "position": {"x": round(final[0], 1), "y": round(final[1], 1), "z": round(final[2], 1)},
            }
        return {"status": "error", "message": f"Position ({x:.0f}, {y:.0f}, {z:.0f}) unreachable"}

    def _execute_move_relative(self, args: dict) -> dict:
        dx, dy, dz = args["dx"], args["dy"], args["dz"]
        ok = self._hw.motion.move_relative(dx, dy, dz)
        if ok:
            final = self._hw.motion.get_pose()
            return {
                "status": "success",
                "message": f"Moved relative ({dx:.0f}, {dy:.0f}, {dz:.0f})",
                "position": {"x": round(final[0], 1), "y": round(final[1], 1), "z": round(final[2], 1)},
            }
        return {"status": "error", "message": f"Relative move ({dx:.0f}, {dy:.0f}, {dz:.0f}) failed"}

    def _execute_press_down(self, args: dict) -> dict:
        max_force = args.get("max_force", 50)
        max_dist = args.get("max_distance_mm", 50.0)
        result = self._hw.motion.force_move(
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

    def _execute_force_move(self, args: dict) -> dict:
        dx, dy, dz = args["dx"], args["dy"], args["dz"]
        max_force = args.get("max_force", 50)
        max_dist = args.get("max_distance_mm", 100.0)
        result = self._hw.motion.force_move(
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

    def _execute_trajectory(self, args: dict) -> dict:
        waypoints_raw = args["waypoints"]
        force_threshold = args.get("force_threshold")
        waypoints = [(w[0], w[1], w[2]) for w in waypoints_raw]

        if len(waypoints) < 2:
            return {"status": "error", "message": "At least 2 waypoints required"}

        result = self._hw.motion.follow_trajectory(
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
                + (f" (stopped: contact)" if result.get("contact") else "")
            ),
        }

    def _execute_pose_get(self, args: dict) -> dict:
        x, y, z = self._hw.motion.get_pose()
        return {
            "status": "success",
            "position": {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1)},
            "message": f"Current position: x={x:.1f}, y={y:.1f}, z={z:.1f} (ground-relative)",
        }

    def _execute_move_home(self, args: dict) -> dict:
        self._hw.motion.home()
        time.sleep(1)
        return {"status": "success", "message": "Moved to home position"}

    def _execute_gripper_ctrl(self, args: dict) -> dict:
        angle = max(0, min(90, args["angle"]))
        self._hw.motion.arm.gripper_angle_ctrl(angle, GRIPPER_SPEED, GRIPPER_ACC)
        time.sleep(0.5)
        state = "open" if angle > 45 else "closed"
        return {"status": "success", "gripper_angle": angle, "message": f"Gripper {state} at {angle} degrees"}

    # ── Tool dispatch ──────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> tuple[Any, list[dict] | None, list[str] | None]:
        """Dispatch tool call. Returns (result, image_blocks, web_images)."""
        if name == "look":
            text, images, web = self._execute_look(args)
            return text, images, web
        if name == "detect":
            text, images, web = self._execute_detect(args)
            return text, images, web
        if name == "segment":
            text, images, web = self._execute_segment(args)
            return text, images, web

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
            return {"status": "error", "message": f"Unknown tool: {name}"}, None, None

        try:
            result = executor(args)
            return result, None, None
        except Exception as e:
            return {"status": "error", "message": str(e)}, None, None

    # ── Action formatting ──────────────────────────────────────

    def _format_action(self, name: str, args: dict) -> str:
        """Format a pending action for the confirmation dialog."""
        if name == "goto_pixel":
            px, py = args["pixel_x"], args["pixel_y"]
            z_off = args.get("z_offset_mm", 50)
            arm_desc = ""
            hw = self._hw
            if hw.vision_depth_frame is not None:
                depth_mm = hw.ct.get_depth_at_pixel(hw.vision_depth_frame, px, py)
                if depth_mm is not None:
                    cam_3d = hw.ct.deproject_pixel(px, py, depth_mm=depth_mm)
                    if cam_3d is not None:
                        arm_3d = hw.ct.camera_to_arm(cam_3d)
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

    # ── Main agent loop ────────────────────────────────────────

    def _execute_task_sync(self, task: str, auto_confirm: bool) -> str:
        """Synchronous agent loop using agent_v3's LLM provider abstraction."""
        from agent_v3.prompts import create_system_prompt, create_task_prompt

        system_prompt = create_system_prompt(
            has_sam2=self._hw.sam2 is not None,
            has_gemini_vision=self._hw.gemini_vision is not None,
        )
        task_prompt = create_task_prompt(task)

        messages = self.provider.create_messages(system_prompt, task_prompt)
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            if self._cancel_requested:
                return "Task cancelled by user."

            iteration += 1
            self._publish("agent.iteration", {"iteration": iteration, "max_iterations": max_iterations})

            try:
                response = self.provider.chat(messages, self.tool_declarations)
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                raise

            if response.text:
                self._publish("agent.iteration", {
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "message": response.text,
                })

            self.provider.append_assistant_response(messages, response)

            if not response.tool_calls:
                return response.text or "Task completed."

            # Execute tool calls
            movement_executed = False
            pending_image_blocks: list[dict] = []

            for tc in response.tool_calls:
                if self._cancel_requested:
                    return "Task cancelled by user."

                self._publish("agent.tool_call", {
                    "tool_name": tc.name, "args": tc.args, "status": "executing",
                })

                # Confirmation for movement tools
                if tc.name in CONFIRM_TOOLS and not auto_confirm:
                    action_str = self._format_action(tc.name, tc.args)
                    approved = self._wait_for_confirmation(action_str)
                    if not approved:
                        result_str = json.dumps({"status": "cancelled", "message": "User cancelled"})
                        self.provider.append_tool_result(messages, tc, result_str)
                        self._publish("agent.tool_call", {
                            "tool_name": tc.name, "args": tc.args,
                            "status": "completed",
                            "result": {"status": "cancelled", "message": "User cancelled"},
                        })
                        continue

                result, images, web_images = self._execute_tool(tc.name, tc.args)
                result_str = result if isinstance(result, str) else json.dumps(result)

                self.provider.append_tool_result(messages, tc, result_str)

                self._publish("agent.tool_call", {
                    "tool_name": tc.name, "args": tc.args,
                    "status": "completed",
                    "result": result if isinstance(result, dict) else {"text": result},
                    "images": web_images,
                })

                if images:
                    pending_image_blocks.extend(images)

                if tc.name in MOVEMENT_TOOLS:
                    result_dict = result if isinstance(result, dict) else {}
                    if result_dict.get("status") == "success":
                        movement_executed = True

            # Add images as user message via provider abstraction
            if pending_image_blocks:
                encoded_parts = []
                for block in pending_image_blocks:
                    if block["type"] == "image":
                        encoded_parts.append(
                            self.provider.encode_image(block["data"], block["mime_type"])
                        )
                    elif block["type"] == "text":
                        encoded_parts.append(
                            self.provider.encode_text(block["text"])
                        )
                self.provider.append_images(messages, encoded_parts)

            # Post-movement capture
            if movement_executed:
                time.sleep(0.5)
                text_result, image_blocks, _ = self._execute_look({})
                encoded_parts = []
                for block in image_blocks:
                    if block["type"] == "image":
                        encoded_parts.append(
                            self.provider.encode_image(block["data"], block["mime_type"])
                        )
                    elif block["type"] == "text":
                        encoded_parts.append(
                            self.provider.encode_text(block["text"])
                        )
                self.provider.append_images(
                    messages, encoded_parts,
                    text="[Updated view after movement. Confirm task completion or continue.]"
                )

        return "Maximum iterations reached. Task may be incomplete."
