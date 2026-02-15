"""Arm control REST endpoints."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..events import EventBus
from ..hardware import HardwareManager

logger = logging.getLogger("server.routers.arm")
router = APIRouter(prefix="/api/arm", tags=["arm"])


# --- Request/Response Models ---


class MoveRequest(BaseModel):
    x: float
    y: float
    z: float = Field(ge=0, description="Ground-relative Z in mm")


class GotoPixelRequest(BaseModel):
    pixel_x: int = Field(ge=0, le=639)
    pixel_y: int = Field(ge=0, le=479)
    z_offset_mm: float = 50.0


class GripperRequest(BaseModel):
    angle: float = Field(ge=0, le=90)


class PoseResponse(BaseModel):
    x: float
    y: float
    z: float


class MoveResponse(BaseModel):
    status: str
    message: str
    position: PoseResponse | None = None


class GripperResponse(BaseModel):
    status: str
    angle: float
    state: str


class ArmStatusResponse(BaseModel):
    connected: bool
    ground_calibrated: bool


# --- Endpoints ---


def _get_hw_bus(request: Request) -> tuple[HardwareManager, EventBus]:
    return request.app.state.hardware, request.app.state.event_bus


@router.get("/status", response_model=ArmStatusResponse)
async def arm_status(request: Request):
    hw, _ = _get_hw_bus(request)
    return ArmStatusResponse(
        connected=hw.motion is not None,
        ground_calibrated=hw.motion is not None and hw.motion._ground_z is not None
        if hw.motion
        else False,
    )


@router.get("/pose", response_model=PoseResponse)
async def get_pose(request: Request):
    hw, bus = _get_hw_bus(request)
    if hw.motion is None:
        raise HTTPException(503, "Arm not connected")

    x, y, z = await hw.run_in_hw_thread(hw.motion.get_pose)
    pose = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
    await bus.publish("arm.position", pose.model_dump())
    return pose


@router.post("/move", response_model=MoveResponse)
async def move_to(body: MoveRequest, request: Request):
    hw, bus = _get_hw_bus(request)
    if hw.motion is None:
        raise HTTPException(503, "Arm not connected")

    async with hw.arm_lock:
        await bus.publish(
            "arm.moving",
            {"target_x": body.x, "target_y": body.y, "target_z": body.z, "status": "started"},
        )

        ok = await hw.run_in_hw_thread(hw.motion.move_to, body.x, body.y, body.z)

        x, y, z = await hw.run_in_hw_thread(hw.motion.get_pose)
        position = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))

        status = "success" if ok else "error"
        await bus.publish(
            "arm.moving",
            {
                "target_x": body.x,
                "target_y": body.y,
                "target_z": body.z,
                "status": "reached" if ok else "failed",
            },
        )
        await bus.publish("arm.position", position.model_dump())

        return MoveResponse(
            status=status,
            message=f"Moved to ({body.x}, {body.y}, {body.z})" if ok else "Move failed",
            position=position,
        )


@router.post("/home", response_model=MoveResponse)
async def move_home(request: Request):
    hw, bus = _get_hw_bus(request)
    if hw.motion is None:
        raise HTTPException(503, "Arm not connected")

    async with hw.arm_lock:
        await hw.run_in_hw_thread(hw.motion.home)
        await asyncio.sleep(1)
        x, y, z = await hw.run_in_hw_thread(hw.motion.get_pose)
        position = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
        await bus.publish("arm.position", position.model_dump())
        return MoveResponse(status="success", message="Moved to home position", position=position)


@router.post("/stop", response_model=MoveResponse)
async def emergency_stop(request: Request):
    """Emergency stop: read current position and hold."""
    hw, bus = _get_hw_bus(request)
    if hw.motion is None:
        raise HTTPException(503, "Arm not connected")

    # Get current position and command arm to stay there
    x, y, z = await hw.run_in_hw_thread(hw.motion.get_pose)
    position = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
    await bus.publish("arm.position", position.model_dump())
    return MoveResponse(status="success", message="Emergency stop — holding position", position=position)


@router.post("/gripper", response_model=GripperResponse)
async def gripper_control(body: GripperRequest, request: Request):
    hw, bus = _get_hw_bus(request)
    if hw.motion is None:
        raise HTTPException(503, "Arm not connected")

    angle = max(0.0, min(90.0, body.angle))

    async with hw.arm_lock:
        await hw.run_in_hw_thread(
            hw.motion.arm.gripper_angle_ctrl, angle, 100, 50
        )
        await asyncio.sleep(0.5)

    state = "open" if angle > 60 else ("closed" if angle < 10 else "partial")
    await bus.publish("arm.gripper", {"angle": angle, "state": state})
    return GripperResponse(status="success", angle=angle, state=state)


@router.post("/goto-pixel", response_model=MoveResponse)
async def goto_pixel(body: GotoPixelRequest, request: Request):
    """Move arm to 3D position from pixel coordinates."""
    hw, bus = _get_hw_bus(request)
    if hw.motion is None or hw.ct is None:
        raise HTTPException(503, "Arm or coordinate transform not ready")

    depth_frame = hw.vision_depth_frame
    if depth_frame is None:
        bundle = hw.latest_frame
        if bundle is None:
            raise HTTPException(400, "No frame available. Call /api/vision/look first.")
        depth_frame = bundle.depth_frame

    # Compute arm coordinates
    depth_mm = hw.ct.get_depth_at_pixel(depth_frame, body.pixel_x, body.pixel_y)
    if depth_mm is None or depth_mm <= 0:
        raise HTTPException(400, f"No valid depth at pixel ({body.pixel_x}, {body.pixel_y})")

    cam_3d = hw.ct.deproject_pixel(body.pixel_x, body.pixel_y, depth_mm=depth_mm)
    if cam_3d is None:
        raise HTTPException(400, "Deprojection failed")

    arm_3d = hw.ct.camera_to_arm(cam_3d)
    if arm_3d is None:
        raise HTTPException(400, "Camera-to-arm transform failed (calibration loaded?)")

    target_x = float(arm_3d[0])
    target_y = float(arm_3d[1])
    target_z = float(arm_3d[2]) + body.z_offset_mm

    async with hw.arm_lock:
        await bus.publish(
            "arm.moving",
            {"target_x": target_x, "target_y": target_y, "target_z": target_z, "status": "started"},
        )
        ok = await hw.run_in_hw_thread(hw.motion.move_to, target_x, target_y, target_z)

        x, y, z = await hw.run_in_hw_thread(hw.motion.get_pose)
        position = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
        await bus.publish("arm.position", position.model_dump())

        return MoveResponse(
            status="success" if ok else "error",
            message=f"Moved to arm ({target_x:.0f}, {target_y:.0f}, {target_z:.0f})" if ok else "Move failed",
            position=position,
        )
