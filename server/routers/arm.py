"""Arm control REST endpoints with per-device targeting."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request
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


class ProbeGroundResponse(BaseModel):
    status: str
    message: str


class ArmStatusResponse(BaseModel):
    connected: bool
    ground_calibrated: bool
    active_device: str | None = None
    connected_devices: list[str] = []


class DevicesResponse(BaseModel):
    devices: list[str]
    active: str | None = None
    connected: list[str] = []


class ConnectRequest(BaseModel):
    device: str


class ConnectResponse(BaseModel):
    status: str
    device: str
    message: str


class ConnectAndProbeAllResponse(BaseModel):
    status: str
    connected: list[str]
    probed: list[str]
    failed: list[str]
    message: str


# --- Helpers ---


def _get_hw_bus(request: Request) -> tuple[HardwareManager, EventBus]:
    return request.app.state.hardware, request.app.state.event_bus


def _resolve_arm(hw: HardwareManager, device: str | None):
    """Resolve device name and get Motion instance, or raise HTTP error."""
    dev = device or hw.active_arm_device
    if dev is None:
        raise HTTPException(503, "No arm connected")
    motion = hw.get_motion(dev)
    if motion is None:
        raise HTTPException(503, f"Arm {dev} not connected")
    return dev, motion


# --- Endpoints ---


@router.get("/devices", response_model=DevicesResponse)
async def list_devices(request: Request):
    """List available arm devices and which are connected."""
    hw, _ = _get_hw_bus(request)
    return DevicesResponse(
        devices=hw.arm_devices,
        active=hw.active_arm_device,
        connected=hw.connected_devices,
    )


@router.post("/connect", response_model=ConnectResponse)
async def connect_arm(body: ConnectRequest, request: Request):
    """Connect to a specific arm device (keeps other arms connected)."""
    hw, bus = _get_hw_bus(request)
    if body.device not in hw.arm_devices:
        raise HTTPException(
            400, f"Unknown device: {body.device}. Available: {hw.arm_devices}"
        )
    try:
        await hw.connect_arm(body.device)
    except Exception as e:
        raise HTTPException(500, f"Failed to connect to {body.device}: {e}")
    await bus.publish("arm.connected", {"device": body.device})
    return ConnectResponse(
        status="success", device=body.device, message=f"Connected to {body.device}"
    )


@router.post("/connect-and-probe-all", response_model=ConnectAndProbeAllResponse)
async def connect_and_probe_all(request: Request):
    """Connect to all available arm devices and probe ground for each one."""
    hw, bus = _get_hw_bus(request)
    connected: list[str] = []
    probed: list[str] = []
    failed: list[str] = []

    for device in hw.arm_devices:
        # Connect
        try:
            await hw.connect_arm(device)
            connected.append(device)
            await bus.publish("arm.connected", {"device": device})
        except Exception as e:
            logger.warning(f"Failed to connect {device}: {e}")
            failed.append(device)
            continue

        # Probe ground
        motion = hw.get_motion(device)
        if motion is None:
            failed.append(device)
            continue

        lock = hw.get_arm_lock(device)
        try:
            async with lock:
                await hw.run_in_hw_thread(motion.probe_ground)
                await hw.run_in_hw_thread(motion.home)
                await asyncio.sleep(1)
                x, y, z = await hw.run_in_hw_thread(motion.get_pose)
                pose = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
                await bus.publish(
                    "arm.position", {"device": device, **pose.model_dump()}
                )
                await bus.publish(
                    "arm.ground_calibrated", {"status": True, "device": device}
                )
            probed.append(device)
        except Exception as e:
            logger.warning(f"Failed to probe ground on {device}: {e}")
            failed.append(device)

    status = "success" if not failed else ("partial" if connected else "error")
    return ConnectAndProbeAllResponse(
        status=status,
        connected=connected,
        probed=probed,
        failed=failed,
        message=f"Connected {len(connected)}, probed {len(probed)} arms"
        + (f" ({len(failed)} failed)" if failed else ""),
    )


@router.post("/disconnect", response_model=ConnectResponse)
async def disconnect_arm(request: Request, device: str | None = Query(None)):
    """Disconnect a specific arm (defaults to active arm)."""
    hw, bus = _get_hw_bus(request)
    dev = device or hw.active_arm_device
    if dev is None:
        raise HTTPException(400, "No arm connected")
    if dev not in hw.connected_devices:
        raise HTTPException(400, f"Arm {dev} is not connected")
    await hw.disconnect_arm(dev)
    await bus.publish("arm.disconnected", {"device": dev})
    return ConnectResponse(
        status="success", device=dev, message=f"Disconnected from {dev}"
    )


@router.get("/status", response_model=ArmStatusResponse)
async def arm_status(request: Request, device: str | None = Query(None)):
    hw, _ = _get_hw_bus(request)
    dev = device or hw.active_arm_device
    motion = hw.get_motion(dev) if dev else None
    return ArmStatusResponse(
        connected=motion is not None,
        ground_calibrated=motion is not None and motion._ground_z is not None,
        active_device=hw.active_arm_device,
        connected_devices=hw.connected_devices,
    )


@router.post("/probe-ground", response_model=ProbeGroundResponse)
async def probe_ground(request: Request, device: str | None = Query(None)):
    """Probe ground to calibrate Z=0. Must be done before ground-relative moves."""
    hw, bus = _get_hw_bus(request)
    dev, motion = _resolve_arm(hw, device)

    lock = hw.get_arm_lock(dev)
    async with lock:
        await hw.run_in_hw_thread(motion.probe_ground)
        await hw.run_in_hw_thread(motion.home)
        await asyncio.sleep(1)
        x, y, z = await hw.run_in_hw_thread(motion.get_pose)
        pose = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
        await bus.publish("arm.position", {"device": dev, **pose.model_dump()})
        await bus.publish("arm.ground_calibrated", {"status": True, "device": dev})

    return ProbeGroundResponse(
        status="success", message="Ground probed and Z=0 calibrated"
    )


@router.get("/pose", response_model=PoseResponse)
async def get_pose(request: Request, device: str | None = Query(None)):
    hw, bus = _get_hw_bus(request)
    dev, motion = _resolve_arm(hw, device)

    x, y, z = await hw.run_in_hw_thread(motion.get_pose)
    pose = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
    await bus.publish("arm.position", {"device": dev, **pose.model_dump()})
    return pose


@router.post("/move", response_model=MoveResponse)
async def move_to(
    body: MoveRequest, request: Request, device: str | None = Query(None)
):
    hw, bus = _get_hw_bus(request)
    dev, motion = _resolve_arm(hw, device)

    lock = hw.get_arm_lock(dev)
    async with lock:
        await bus.publish(
            "arm.moving",
            {
                "device": dev,
                "target_x": body.x,
                "target_y": body.y,
                "target_z": body.z,
                "status": "started",
            },
        )

        ok = await hw.run_in_hw_thread(motion.move_to, body.x, body.y, body.z)

        x, y, z = await hw.run_in_hw_thread(motion.get_pose)
        position = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))

        status = "success" if ok else "error"
        await bus.publish(
            "arm.moving",
            {
                "device": dev,
                "target_x": body.x,
                "target_y": body.y,
                "target_z": body.z,
                "status": "reached" if ok else "failed",
            },
        )
        await bus.publish("arm.position", {"device": dev, **position.model_dump()})

        return MoveResponse(
            status=status,
            message=f"Moved to ({body.x}, {body.y}, {body.z})" if ok else "Move failed",
            position=position,
        )


@router.post("/home", response_model=MoveResponse)
async def move_home(request: Request, device: str | None = Query(None)):
    hw, bus = _get_hw_bus(request)
    dev, motion = _resolve_arm(hw, device)

    lock = hw.get_arm_lock(dev)
    async with lock:
        await hw.run_in_hw_thread(motion.home)
        await asyncio.sleep(1)
        x, y, z = await hw.run_in_hw_thread(motion.get_pose)
        position = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
        await bus.publish("arm.position", {"device": dev, **position.model_dump()})
        return MoveResponse(
            status="success", message="Moved to home position", position=position
        )


@router.post("/stop", response_model=MoveResponse)
async def emergency_stop(request: Request):
    """Emergency stop: freeze ALL connected arms at their current positions.

    Bypasses the hw executor thread so it works even while a move is in progress.
    """
    hw, bus = _get_hw_bus(request)
    if not hw.connected_devices:
        raise HTTPException(503, "No arm connected")

    # Stop all connected arms in parallel threads (bypass busy hw executor)
    loop = asyncio.get_event_loop()
    devices = list(hw.connected_devices)
    tasks = []
    for dev in devices:
        tasks.append(loop.run_in_executor(None, _estop_one, hw, dev))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Publish position updates for each stopped arm
    last_position = None
    for dev, result in zip(devices, results):
        if isinstance(result, PoseResponse):
            await bus.publish("arm.position", {"device": dev, **result.model_dump()})
            last_position = result
        else:
            logger.error(f"E-stop failed for {dev}: {result}")

    return MoveResponse(
        status="success",
        message=f"Emergency stop — {len(devices)} arm(s) frozen",
        position=last_position,
    )


def _estop_one(hw: HardwareManager, device: str) -> PoseResponse:
    """Read current joints and re-command them to freeze one arm (runs in thread)."""
    from motion_controller.motion import fk

    motion = hw.get_motion(device)
    if motion is None:
        raise ValueError(f"Arm {device} not connected")
    arm = motion.arm
    fb = arm.feedback_get()
    # Command arm to hold at its current joint angles, interrupting any in-progress move
    arm.joints_radian_ctrl(radians=[fb[3], fb[4], fb[5], fb[6]], speed=4000, acc=200)
    # Compute ground-relative position from joint angles
    x, y, z = fk(fb[3], fb[4], fb[5])
    if motion._ground_z is not None:
        z = z - motion._ground_z
    return PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))


@router.post("/gripper", response_model=GripperResponse)
async def gripper_control(
    body: GripperRequest, request: Request, device: str | None = Query(None)
):
    hw, bus = _get_hw_bus(request)
    dev, motion = _resolve_arm(hw, device)

    angle = max(0.0, min(90.0, body.angle))

    lock = hw.get_arm_lock(dev)
    async with lock:
        await hw.run_in_hw_thread(motion.arm.gripper_angle_ctrl, angle, 100, 50)
        await asyncio.sleep(0.5)

    state = "open" if angle > 60 else ("closed" if angle < 10 else "partial")
    await bus.publish("arm.gripper", {"device": dev, "angle": angle, "state": state})
    return GripperResponse(status="success", angle=angle, state=state)


@router.post("/goto-pixel", response_model=MoveResponse)
async def goto_pixel(
    body: GotoPixelRequest, request: Request, device: str | None = Query(None)
):
    """Move arm to 3D position from pixel coordinates."""
    hw, bus = _get_hw_bus(request)
    dev, motion = _resolve_arm(hw, device)

    if hw.ct is None:
        raise HTTPException(503, "Coordinate transform not ready")

    depth_frame = hw.vision_depth_frame
    if depth_frame is None:
        bundle = hw.latest_frame
        if bundle is None:
            raise HTTPException(400, "No frame available. Call /api/vision/look first.")
        depth_frame = bundle.depth_frame

    # Compute arm coordinates
    depth_mm = hw.ct.get_depth_at_pixel(depth_frame, body.pixel_x, body.pixel_y)
    if depth_mm is None or depth_mm <= 0:
        raise HTTPException(
            400, f"No valid depth at pixel ({body.pixel_x}, {body.pixel_y})"
        )

    cam_3d = hw.ct.deproject_pixel(body.pixel_x, body.pixel_y, depth_mm=depth_mm)
    if cam_3d is None:
        raise HTTPException(400, "Deprojection failed")

    arm_3d = hw.ct.camera_to_arm(cam_3d)
    if arm_3d is None:
        raise HTTPException(400, "Camera-to-arm transform failed (calibration loaded?)")

    target_x = float(arm_3d[0])
    target_y = float(arm_3d[1])
    target_z = float(arm_3d[2]) + body.z_offset_mm

    lock = hw.get_arm_lock(dev)
    async with lock:
        await bus.publish(
            "arm.moving",
            {
                "device": dev,
                "target_x": target_x,
                "target_y": target_y,
                "target_z": target_z,
                "status": "started",
            },
        )
        ok = await hw.run_in_hw_thread(motion.move_to, target_x, target_y, target_z)

        x, y, z = await hw.run_in_hw_thread(motion.get_pose)
        position = PoseResponse(x=round(x, 1), y=round(y, 1), z=round(z, 1))
        await bus.publish("arm.position", {"device": dev, **position.model_dump()})

        return MoveResponse(
            status="success" if ok else "error",
            message=f"Moved to arm ({target_x:.0f}, {target_y:.0f}, {target_z:.0f})"
            if ok
            else "Move failed",
            position=position,
        )
