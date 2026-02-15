"""WebAgentV2: wraps the AgentV2 loop with event emission for web UI."""

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


class WebAgentV2:
    """Wraps AgentV2's process_task loop to emit events via EventBus."""

    def __init__(
        self,
        hw: HardwareManager,
        bus: EventBus,
        openai_api_key: str,
        helicone_api_key: str,
        model: str = "gpt-5.2",
        reasoning_effort: str = "low",
    ):
        self._hw = hw
        self._bus = bus
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._state = "idle"  # idle | running | awaiting_confirm | error
        self._current_task: str | None = None
        self._cancel_requested = False
        self._pending_confirmation: asyncio.Future | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # Create OpenAI client
        from agent.vlm_agent import create_openai_client

        self._client = create_openai_client(
            openai_api_key, helicone_api_key, debug=False
        )

        # Build tool declarations
        from agent_v2.tools import TOOL_DECLARATIONS

        self._tools = []
        for decl in TOOL_DECLARATIONS:
            self._tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": decl["name"],
                        "description": decl["description"],
                        "parameters": decl["parameters"],
                    },
                }
            )

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_task(self) -> str | None:
        return self._current_task

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

    def _publish(self, event_type: str, payload: dict):
        """Thread-safe publish from sync context."""
        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._bus.publish(event_type, payload), self._event_loop
            )

    def _encode_image(self, image: np.ndarray, quality: int = 85) -> str:
        _, jpeg = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return base64.b64encode(jpeg.tobytes()).decode("utf-8")

    def _execute_look(self) -> tuple[str, list[dict], list[str]]:
        """Capture frame + encode for SAM2. Returns (text, openai_content, web_images)."""
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

        # Cache for segment/goto
        hw.vision_color = color_image
        hw.vision_depth_frame = depth_frame

        # Encode for SAM2
        encode_time = hw.sam2.set_image(color_image)

        h, w = color_image.shape[:2]
        text_result = (
            f"Camera frame captured ({w}x{h}). "
            f"SAM2 image encoded in {encode_time:.1f}s. "
            f"Ready for segment() queries."
        )

        # Draw grid on copy
        from agent_v2.agent import AgentV2

        color_with_grid = AgentV2._draw_coordinate_grid(color_image)
        color_b64 = self._encode_image(color_with_grid)
        depth_b64 = self._encode_image(depth_image)

        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{color_b64}", "detail": "high"},
            },
            {"type": "text", "text": "[Camera image with coordinate grid. Depth image below.]"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{depth_b64}", "detail": "high"},
            },
        ]

        web_images = [
            f"data:image/jpeg;base64,{color_b64}",
            f"data:image/jpeg;base64,{depth_b64}",
        ]

        return text_result, image_content, web_images

    def _execute_segment(self, args: dict) -> tuple[str, list[dict], list[str]]:
        """SAM2 segmentation. Returns (text, openai_content, web_images)."""
        hw = self._hw
        pixel_x = args["pixel_x"]
        pixel_y = args["pixel_y"]

        if hw.vision_color is None:
            return json.dumps({"status": "error", "message": "Call look() first"}), [], []

        mask, score, bbox = hw.sam2.segment_point(pixel_x, pixel_y)

        arm_coords = None
        depth_mm = hw.ct.get_depth_at_pixel(hw.vision_depth_frame, pixel_x, pixel_y)
        if depth_mm is not None:
            cam_3d = hw.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
            if cam_3d is not None:
                arm_3d = hw.ct.camera_to_arm(cam_3d)
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
        }

        # Annotated image
        annotated = hw.vision_color.copy()
        overlay = annotated.copy()
        overlay[mask] = [0, 200, 0]
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
        cv2.rectangle(annotated, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.drawMarker(annotated, (pixel_x, pixel_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        annotated_b64 = self._encode_image(annotated)

        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{annotated_b64}", "detail": "high"},
            },
            {"type": "text", "text": f"[Segmentation at ({pixel_x},{pixel_y}), score={score:.3f}]"},
        ]
        web_images = [f"data:image/jpeg;base64,{annotated_b64}"]

        return json.dumps(result), image_content, web_images

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
                "arm_position": {"x": round(tx, 1), "y": round(ty, 1), "z": round(tz, 1)},
            }
        return {"status": "error", "message": f"Move failed — ({tx:.0f},{ty:.0f},{tz:.0f}) may be unreachable"}

    def _execute_pose_get(self) -> dict:
        x, y, z = self._hw.motion.get_pose()
        return {"status": "success", "position": {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1)}}

    def _execute_move_home(self) -> dict:
        self._hw.motion.home()
        time.sleep(1)
        return {"status": "success", "message": "Moved to home position"}

    def _execute_gripper_ctrl(self, args: dict) -> dict:
        angle = max(0, min(90, args["angle"]))
        self._hw.motion.arm.gripper_angle_ctrl(angle, GRIPPER_SPEED, GRIPPER_ACC)
        time.sleep(0.5)
        state = "open" if angle > 45 else "closed"
        return {"status": "success", "gripper_angle": angle, "message": f"Gripper {state} at {angle} degrees"}

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

    async def confirm_action(self, approved: bool):
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(approved)

    async def cancel(self):
        self._cancel_requested = True
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(False)

    def _execute_task_sync(self, task: str, auto_confirm: bool) -> str:
        """Synchronous agent loop (runs in thread)."""
        from agent_v2.prompts import SYSTEM_PROMPT, create_task_prompt

        task_prompt = create_task_prompt(task)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            if self._cancel_requested:
                return "Task cancelled by user."

            iteration += 1
            self._publish("agent.iteration", {"iteration": iteration, "max_iterations": max_iterations})

            # LLM call
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": self._tools,
                "tool_choice": "auto",
            }
            if "o1" in self._model.lower() or "o3" in self._model.lower():
                request_kwargs["extra_body"] = {"reasoning_effort": self._reasoning_effort}

            response = self._client.chat.completions.create(**request_kwargs)

            if not response.choices:
                return "Model returned empty response."

            choice = response.choices[0]
            assistant_message = choice.message

            if assistant_message.content:
                self._publish(
                    "agent.iteration",
                    {"iteration": iteration, "max_iterations": max_iterations, "message": assistant_message.content},
                )

            messages.append(assistant_message.model_dump())

            if not assistant_message.tool_calls:
                return assistant_message.content or "Task completed."

            # Execute tool calls
            movement_executed = False
            pending_images: list[dict] | None = None

            for tool_call in assistant_message.tool_calls:
                if self._cancel_requested:
                    return "Task cancelled by user."

                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                self._publish(
                    "agent.tool_call",
                    {"tool_name": tool_name, "args": args, "status": "executing"},
                )

                web_images = None
                result: Any
                images: list[dict] | None = None

                if tool_name == "look":
                    text_result, images, web_images = self._execute_look()
                    result = text_result
                elif tool_name == "segment":
                    text_result, images, web_images = self._execute_segment(args)
                    result = text_result
                else:
                    # Movement tools — may need confirmation
                    needs_confirm = tool_name in ("goto_pixel", "move_home", "gripper_ctrl")
                    if needs_confirm and not auto_confirm:
                        if tool_name == "goto_pixel":
                            action_str = f"GOTO pixel ({args.get('pixel_x')},{args.get('pixel_y')})"
                        elif tool_name == "move_home":
                            action_str = "MOVE ARM to HOME"
                        else:
                            state = "OPEN" if args.get("angle", 0) > 45 else "CLOSE"
                            action_str = f"GRIPPER {state} to {args.get('angle')}°"

                        approved = self._wait_for_confirmation(action_str)
                        if not approved:
                            result = {"status": "cancelled", "message": "User cancelled"}
                        else:
                            result = self._execute_movement(tool_name, args)
                    else:
                        result = self._execute_movement(tool_name, args)

                result_str = result if isinstance(result, str) else json.dumps(result)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_str})

                self._publish(
                    "agent.tool_call",
                    {
                        "tool_name": tool_name,
                        "args": args,
                        "status": "completed",
                        "result": result if isinstance(result, dict) else {"text": result},
                        "images": web_images,
                    },
                )

                if images:
                    pending_images = images

                if tool_name in ("goto_pixel", "move_home", "gripper_ctrl"):
                    result_dict = result if isinstance(result, dict) else {}
                    if result_dict.get("status") == "success":
                        movement_executed = True

            # Add images as user message for LLM
            if pending_images:
                messages.append({"role": "user", "content": pending_images})

            # Post-movement view
            if movement_executed:
                time.sleep(0.5)
                text_result, image_content, _ = self._execute_look()
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            *image_content,
                            {
                                "type": "text",
                                "text": "[Updated view after movement. Confirm completion or continue.]",
                            },
                        ],
                    }
                )

        return "Maximum iterations reached. Task may be incomplete."

    def _execute_movement(self, tool_name: str, args: dict) -> dict:
        if tool_name == "goto_pixel":
            return self._execute_goto_pixel(args)
        elif tool_name == "pose_get":
            return self._execute_pose_get()
        elif tool_name == "move_home":
            return self._execute_move_home()
        elif tool_name == "gripper_ctrl":
            return self._execute_gripper_ctrl(args)
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
