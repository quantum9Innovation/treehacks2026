"""Calibration endpoints for web-based camera-to-arm calibration."""

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..events import EventBus
from ..hardware import HardwareManager

logger = logging.getLogger("server.routers.calibration")
router = APIRouter(prefix="/api/calibration", tags=["calibration"])


class CalibrationStatusResponse(BaseModel):
    has_calibration: bool
    rmse_mm: float | None = None
    session_active: bool = False
    current_step: int = 0
    total_steps: int = 0


class StartCalibrationRequest(BaseModel):
    mode: str = "manual"  # "manual" or "aruco"


class ClickRequest(BaseModel):
    pixel_x: int = Field(ge=0, le=639)
    pixel_y: int = Field(ge=0, le=479)


# Active calibration session (one at a time)
_session: "WebCalibrationSession | None" = None


class WebCalibrationSession:
    """State machine for web-based calibration."""

    def __init__(self, hw: HardwareManager, bus: EventBus):
        from agent_v2.calibration import CALIBRATION_POSITIONS

        self.hw = hw
        self.bus = bus
        self.positions = CALIBRATION_POSITIONS
        self.cam_points: list[np.ndarray] = []
        self.arm_points: list[np.ndarray] = []
        self.current_step = 0
        self.state = "idle"  # idle | moving | waiting_click | done

    @property
    def total_steps(self) -> int:
        return len(self.positions)

    async def start(self):
        self.current_step = 0
        self.cam_points = []
        self.arm_points = []
        await self._move_to_step(0)

    async def _move_to_step(self, step: int):
        self.state = "moving"
        x, y, z = self.positions[step]

        await self.hw.run_in_hw_thread(self.hw.motion.move_to, x, y, z)

        import asyncio
        await asyncio.sleep(1.0)

        ax, ay, az = await self.hw.run_in_hw_thread(self.hw.motion.get_pose)
        self._current_arm_xyz = np.array([ax, ay, az])
        self.state = "waiting_click"

        await self.bus.publish(
            "calibration.progress",
            {
                "step": step + 1,
                "total_steps": self.total_steps,
                "position": {"x": round(ax, 1), "y": round(ay, 1), "z": round(az, 1)},
                "status": "waiting_for_click",
                "point_count": len(self.cam_points),
            },
        )

    async def record_click(self, pixel_x: int, pixel_y: int) -> dict:
        if self.state != "waiting_click":
            return {"status": "error", "message": "Not waiting for a click"}

        # Get depth frame
        depth_frame = self.hw.vision_depth_frame
        if depth_frame is None:
            bundle = self.hw.latest_frame
            if bundle is None:
                return {"status": "error", "message": "No frame available"}
            depth_frame = bundle.depth_frame

        depth_mm = self.hw.ct.get_depth_at_pixel(depth_frame, pixel_x, pixel_y)
        if depth_mm is None or depth_mm <= 0:
            return {"status": "error", "message": f"No valid depth at ({pixel_x}, {pixel_y})"}

        cam_3d = self.hw.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
        if cam_3d is None:
            return {"status": "error", "message": "Deprojection failed"}

        self.cam_points.append(np.array(cam_3d))
        self.arm_points.append(self._current_arm_xyz.copy())

        self.current_step += 1
        if self.current_step >= self.total_steps:
            return await self._solve()

        await self._move_to_step(self.current_step)
        return {"status": "recorded", "point_count": len(self.cam_points)}

    async def skip(self) -> dict:
        self.current_step += 1
        if self.current_step >= self.total_steps:
            if len(self.cam_points) >= 4:
                return await self._solve()
            return {"status": "error", "message": "Need at least 4 points"}

        await self._move_to_step(self.current_step)
        return {"status": "skipped", "point_count": len(self.cam_points)}

    async def _solve(self) -> dict:
        from agent_v2.calibration import calibration_path_for_device, save_calibration, solve_affine_transform

        cam_arr = np.array(self.cam_points)
        arm_arr = np.array(self.arm_points)
        M, rmse = solve_affine_transform(cam_arr, arm_arr)

        # Save to per-arm calibration file
        base_path = Path(self.hw._calibration_path)
        if self.hw.active_arm_device:
            save_path = calibration_path_for_device(base_path, self.hw.active_arm_device)
        else:
            save_path = base_path
        save_calibration(M, rmse, save_path)

        # Reload into coordinate transform
        self.hw.ct._M = M

        quality = "excellent" if rmse < 10 else ("good" if rmse < 20 else "poor")
        self.state = "done"

        result = {
            "status": "done",
            "rmse_mm": round(float(rmse), 2),
            "quality": quality,
            "points_used": len(self.cam_points),
        }
        await self.bus.publish("calibration.result", result)
        return result


@router.get("/status", response_model=CalibrationStatusResponse)
async def calibration_status(request: Request):
    hw: HardwareManager = request.app.state.hardware
    global _session

    rmse = None
    if hw.ct is not None and hw.ct._M is not None:
        # Try to read RMSE from per-arm calibration file
        try:
            import json
            cal_path_str = hw.active_calibration_path
            if cal_path_str:
                cal_path = Path(cal_path_str)
                if cal_path.exists():
                    data = json.loads(cal_path.read_text())
                    rmse = data.get("rmse_mm")
        except Exception:
            pass

    return CalibrationStatusResponse(
        has_calibration=hw.has_calibration,
        rmse_mm=rmse,
        session_active=_session is not None and _session.state not in ("idle", "done"),
        current_step=_session.current_step if _session else 0,
        total_steps=_session.total_steps if _session else 0,
    )


@router.post("/start")
async def start_calibration(body: StartCalibrationRequest, request: Request):
    hw: HardwareManager = request.app.state.hardware
    bus: EventBus = request.app.state.event_bus
    global _session

    if hw.motion is None or hw.ct is None:
        raise HTTPException(503, "Hardware not ready")

    _session = WebCalibrationSession(hw, bus)

    async with hw.arm_lock:
        await _session.start()

    return {
        "status": "started",
        "total_steps": _session.total_steps,
    }


@router.post("/click")
async def record_click(body: ClickRequest, request: Request):
    hw: HardwareManager = request.app.state.hardware
    global _session

    if _session is None:
        raise HTTPException(400, "No calibration session active")

    async with hw.arm_lock:
        result = await _session.record_click(body.pixel_x, body.pixel_y)

    if result.get("status") == "done":
        _session = None

    return result


@router.post("/skip")
async def skip_point(request: Request):
    hw: HardwareManager = request.app.state.hardware
    global _session

    if _session is None:
        raise HTTPException(400, "No calibration session active")

    async with hw.arm_lock:
        result = await _session.skip()

    if result.get("status") == "done":
        _session = None

    return result


@router.post("/abort")
async def abort_calibration(request: Request):
    global _session

    if _session is None:
        raise HTTPException(400, "No calibration session active")

    points = len(_session.cam_points)
    _session = None
    return {"status": "aborted", "points_collected": points}
