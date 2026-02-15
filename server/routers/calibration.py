"""Calibration endpoints for web-based camera-to-arm calibration."""

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..events import EventBus
from ..hardware import HardwareManager

logger = logging.getLogger("server.routers.calibration")
router = APIRouter(prefix="/api/calibration", tags=["calibration"])

# Hardcoded arm-to-marker mapping for multi-arm calibration
ARM_MARKER_MAP: dict[str, int] = {
    "/dev/ARM1": 0,
    "/dev/ARM2": 1,
    "/dev/ARM3": 2,
}


class CalibrationStatusResponse(BaseModel):
    has_calibration: bool
    rmse_mm: float | None = None
    session_active: bool = False
    current_step: int = 0
    total_steps: int = 0
    mode: str = "manual"


class StartCalibrationRequest(BaseModel):
    mode: str = "manual"  # "manual", "aruco", or "aruco_all"
    marker_id: int | None = None


class ClickRequest(BaseModel):
    pixel_x: int = Field(ge=0, le=639)
    pixel_y: int = Field(ge=0, le=479)


# Active calibration session (one at a time)
_session: "WebCalibrationSession | ArucoCalibrationSession | MultiArmArucoCalibrationSession | None" = None


def _solve_and_save(
    hw: HardwareManager,
    cam_points: list[np.ndarray],
    arm_points: list[np.ndarray],
    device: str | None = None,
) -> dict:
    """Solve affine transform, save calibration, return result dict (no WS publish)."""
    from agent_v2.calibration import (
        calibration_path_for_device,
        save_calibration,
        solve_affine_transform,
    )

    cam_arr = np.array(cam_points)
    arm_arr = np.array(arm_points)
    M, rmse = solve_affine_transform(cam_arr, arm_arr)

    # Save to per-arm calibration file
    base_path = Path(hw._calibration_path)
    dev = device or hw.active_arm_device
    if dev:
        save_path = calibration_path_for_device(base_path, dev)
    else:
        save_path = base_path
    save_calibration(M, rmse, save_path)

    # Reload into coordinate transform
    hw.ct._M = M

    quality = "excellent" if rmse < 10 else ("good" if rmse < 20 else "poor")

    return {
        "status": "done",
        "rmse_mm": round(float(rmse), 2),
        "quality": quality,
        "points_used": len(cam_points),
    }


async def _solve_calibration(
    hw: HardwareManager,
    bus: EventBus,
    cam_points: list[np.ndarray],
    arm_points: list[np.ndarray],
) -> dict:
    """Solve affine transform from collected point pairs, save, and publish result."""
    result = _solve_and_save(hw, cam_points, arm_points)
    await bus.publish("calibration.result", result)
    return result


def _draw_aruco_debug_overlay(
    color_image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    target_marker_id: int | None,
    arm_label: str,
    step: int,
    total_steps: int,
    settle_count: int,
    settle_target: int,
    status: str,
) -> None:
    """Draw rich ArUco debug info on the live camera feed.

    - All detected markers: green outlines + ID labels
    - Target marker: yellow crosshair + coordinates
    - Status bar at top with arm/step/state info
    - Rejected candidates: red outlines
    """
    corners, ids, rejected = detector.detectMarkers(color_image)

    # Draw all detected markers (green outlines + IDs)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        # Extra: put marker ID text near each marker center
        for i, mid in enumerate(ids.flatten()):
            center = corners[i][0].mean(axis=0).astype(int)
            cv2.putText(
                color_image,
                f"ID={mid}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # Highlight target marker with yellow crosshair
    if target_marker_id is not None and ids is not None and len(ids) > 0:
        for i, mid in enumerate(ids.flatten()):
            if mid == target_marker_id:
                cx, cy = corners[i][0].mean(axis=0).astype(int)
                # Yellow crosshair circle
                cv2.circle(color_image, (cx, cy), 20, (0, 255, 255), 2)
                cv2.line(color_image, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 1)
                cv2.line(color_image, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 1)
                # Target label + coords
                cv2.putText(
                    color_image,
                    f"TARGET ID={target_marker_id} ({cx},{cy})",
                    (cx + 25, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                break

    # Draw rejected candidates as red outlines
    if rejected and len(rejected) > 0:
        for rej_corners in rejected:
            pts = rej_corners[0].astype(int)
            for j in range(4):
                cv2.line(
                    color_image,
                    tuple(pts[j]),
                    tuple(pts[(j + 1) % 4]),
                    (0, 0, 255),
                    1,
                )

    # Status bar at top
    marker_str = (
        f"marker {target_marker_id}" if target_marker_id is not None else "any marker"
    )
    if status == "detecting":
        state_str = f"Detecting {marker_str} | Settle {settle_count}/{settle_target}"
    elif status == "moving":
        state_str = "Moving to position"
    else:
        state_str = status

    bar_text = f"{arm_label} | Pos {step}/{total_steps} | {state_str}"

    # Semi-transparent background bar
    cv2.rectangle(color_image, (0, 0), (640, 24), (0, 0, 0), -1)
    cv2.putText(
        color_image,
        bar_text,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


class WebCalibrationSession:
    """State machine for web-based manual calibration."""

    mode = "manual"

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
            return {
                "status": "error",
                "message": f"No valid depth at ({pixel_x}, {pixel_y})",
            }

        cam_3d = self.hw.ct.deproject_pixel(pixel_x, pixel_y, depth_mm=depth_mm)
        if cam_3d is None:
            return {"status": "error", "message": "Deprojection failed"}

        self.cam_points.append(np.array(cam_3d))
        self.arm_points.append(self._current_arm_xyz.copy())

        self.current_step += 1
        if self.current_step >= self.total_steps:
            result = await _solve_calibration(
                self.hw, self.bus, self.cam_points, self.arm_points
            )
            self.state = "done"
            return result

        await self._move_to_step(self.current_step)
        return {"status": "recorded", "point_count": len(self.cam_points)}

    async def skip(self) -> dict:
        self.current_step += 1
        if self.current_step >= self.total_steps:
            if len(self.cam_points) >= 4:
                result = await _solve_calibration(
                    self.hw, self.bus, self.cam_points, self.arm_points
                )
                self.state = "done"
                return result
            return {"status": "error", "message": "Need at least 4 points"}

        await self._move_to_step(self.current_step)
        return {"status": "skipped", "point_count": len(self.cam_points)}


def _create_aruco_detector() -> cv2.aruco.ArucoDetector:
    """Create an ArUco detector with tuned params for small markers on arm tip."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 4
    aruco_params.minMarkerPerimeterRate = 0.02
    aruco_params.errorCorrectionRate = 0.8
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def _find_target_marker(
    corners: object,
    ids: np.ndarray | None,
    marker_id: int | None,
) -> tuple[int, np.ndarray] | None:
    """Pick the target marker from detection results.

    Returns (marker_id, center_xy) or None.
    """
    if ids is None or len(ids) == 0:
        return None
    if marker_id is not None:
        for i, mid in enumerate(ids.flatten()):
            if mid == marker_id:
                center = corners[i][0].mean(axis=0)
                return int(mid), center
        return None
    center = corners[0][0].mean(axis=0)
    return int(ids[0][0]), center


class ArucoCalibrationSession:
    """Self-driving calibration using ArUco marker detection.

    Runs as a background asyncio.Task. The arm moves to each calibration
    position and ArUco markers are detected automatically from the camera
    feed. A frame_annotator is set on hw to draw detected markers on the
    WebRTC stream for live visualization.
    """

    mode = "aruco"
    SETTLE_FRAMES = 5
    MAX_ATTEMPTS = 150  # ~5s at 30fps

    def __init__(
        self,
        hw: HardwareManager,
        bus: EventBus,
        marker_id: int | None = None,
    ):
        from agent_v2.calibration import CALIBRATION_POSITIONS

        self.hw = hw
        self.bus = bus
        self.marker_id = marker_id
        self.positions = CALIBRATION_POSITIONS
        self.cam_points: list[np.ndarray] = []
        self.arm_points: list[np.ndarray] = []
        self.current_step = 0
        self.state = "idle"

        self._skip_flag = False
        self._abort_flag = False
        self._task: asyncio.Task | None = None

        self._detector = _create_aruco_detector()

        # Annotator overlay state (read by _annotate_frame in camera thread)
        self._ann_arm_label: str = ""
        self._ann_step: int = 0
        self._ann_total: int = 0
        self._ann_settle_count: int = 0
        self._ann_status: str = "idle"

    @property
    def total_steps(self) -> int:
        return len(self.positions)

    def _annotate_frame(self, color_image: np.ndarray) -> None:
        """Frame annotator callback — draws rich ArUco debug overlay."""
        _draw_aruco_debug_overlay(
            color_image,
            self._detector,
            target_marker_id=self.marker_id,
            arm_label=self._ann_arm_label
            or (
                Path(self.hw.active_arm_device).name
                if self.hw.active_arm_device
                else "ARM"
            ),
            step=self._ann_step,
            total_steps=self._ann_total or self.total_steps,
            settle_count=self._ann_settle_count,
            settle_target=self.SETTLE_FRAMES,
            status=self._ann_status,
        )

    async def start(self):
        """Begin the ArUco calibration background task."""
        self.state = "running"
        active = self.hw.active_arm_device
        self._ann_arm_label = Path(active).name if active else "ARM"
        self._ann_total = self.total_steps
        # Set frame annotator for live marker visualization
        self.hw.frame_annotator = self._annotate_frame
        # Home other connected arms to get them out of the way
        await self._home_other_arms()
        # Launch background loop
        self._task = asyncio.create_task(self._run_loop())

    async def _home_other_arms(self):
        """Move all non-active connected arms out of the way."""
        active = self.hw.active_arm_device
        others = [d for d in self.hw.connected_devices if d != active]
        if not others:
            return
        logger.info(f"Moving {len(others)} other arm(s) out of the way")
        for device in others:
            motion = self.hw.get_motion(device)
            if motion is not None:
                lock = self.hw.get_arm_lock(device)
                async with lock:
                    # Retract arm close to base, high up and out of the way
                    await self.hw.run_in_hw_thread(motion.move_to, 150, 0, 200)
                    await asyncio.sleep(1.5)

    async def _run_loop(self):
        """Main calibration loop — runs as a background task."""
        global _session
        try:
            active_device = self.hw.active_arm_device
            if active_device is None:
                logger.error("No active arm device")
                return
            lock = self.hw.get_arm_lock(active_device)

            for i, (x, y, z) in enumerate(self.positions):
                if self._abort_flag:
                    break

                self.current_step = i
                self.state = "moving"
                self._ann_step = i + 1
                self._ann_status = "moving"
                self._ann_settle_count = 0

                logger.info(
                    f"Position {i + 1}/{self.total_steps}: moving to ({x}, {y}, {z})"
                )

                await self.bus.publish(
                    "calibration.progress",
                    {
                        "step": i + 1,
                        "total_steps": self.total_steps,
                        "status": "moving",
                        "point_count": len(self.cam_points),
                    },
                )

                # Move arm
                async with lock:
                    await self.hw.run_in_hw_thread(self.hw.motion.move_to, x, y, z)
                await asyncio.sleep(1.0)

                if self._abort_flag:
                    break

                # Get actual arm position
                async with lock:
                    ax, ay, az = await self.hw.run_in_hw_thread(self.hw.motion.get_pose)
                arm_xyz = np.array([ax, ay, az])

                self.state = "detecting"
                self._ann_status = "detecting"

                # Detection loop
                collected_centers: list[np.ndarray] = []
                attempt = 0
                self._skip_flag = False

                while attempt < self.MAX_ATTEMPTS:
                    if self._abort_flag or self._skip_flag:
                        break

                    bundle = self.hw.latest_frame
                    if bundle is None:
                        await asyncio.sleep(1 / 30)
                        attempt += 1
                        continue

                    # Detect markers on a copy (annotator draws on live feed separately)
                    corners, ids, _ = self._detector.detectMarkers(bundle.raw_color_image)
                    result = _find_target_marker(corners, ids, self.marker_id)

                    if result is not None:
                        _, center = result
                        collected_centers.append(center)
                        self._ann_settle_count = len(collected_centers)
                        logger.info(
                            f"Position {i + 1}/{self.total_steps}: detected marker "
                            f"{result[0]} at pixel ({int(center[0])}, {int(center[1])}), "
                            f"settle {len(collected_centers)}/{self.SETTLE_FRAMES}"
                        )

                    # Publish progress with settle count
                    if (
                        attempt % 10 == 0
                        or len(collected_centers) == self.SETTLE_FRAMES
                    ):
                        await self.bus.publish(
                            "calibration.progress",
                            {
                                "step": i + 1,
                                "total_steps": self.total_steps,
                                "status": "detecting",
                                "point_count": len(self.cam_points),
                                "settle_count": len(collected_centers),
                                "settle_target": self.SETTLE_FRAMES,
                                "position": {
                                    "x": round(ax, 1),
                                    "y": round(ay, 1),
                                    "z": round(az, 1),
                                },
                            },
                        )

                    if len(collected_centers) >= self.SETTLE_FRAMES:
                        break

                    await asyncio.sleep(1 / 30)
                    attempt += 1

                if self._abort_flag:
                    break

                if self._skip_flag:
                    logger.info(f"Position {i + 1} skipped by user")
                    continue

                if len(collected_centers) < self.SETTLE_FRAMES:
                    logger.warning(
                        f"Position {i + 1}: no marker detected after {attempt} frames, skipping"
                    )
                    await self.bus.publish(
                        "calibration.progress",
                        {
                            "step": i + 1,
                            "total_steps": self.total_steps,
                            "status": "timeout",
                            "point_count": len(self.cam_points),
                        },
                    )
                    continue

                # Average marker centers and deproject
                avg_center = np.mean(collected_centers, axis=0)
                px, py = int(round(avg_center[0])), int(round(avg_center[1]))

                cam_3d = self.hw.ct.deproject_patch(
                    px, py, bundle.depth_frame, patch_size=15
                )
                if cam_3d is None:
                    logger.warning(
                        f"Position {i + 1}: no valid depth at ({px}, {py}), skipping"
                    )
                    continue

                self.cam_points.append(cam_3d)
                self.arm_points.append(arm_xyz)
                logger.info(
                    f"Position {i + 1}: averaged center ({px},{py}), "
                    f"cam_3d={cam_3d}, arm={arm_xyz}"
                )

            # Done with all positions — solve if enough points
            if not self._abort_flag and len(self.cam_points) >= 4:
                result = await _solve_calibration(
                    self.hw, self.bus, self.cam_points, self.arm_points
                )
                self.state = "done"
                logger.info(f"ArUco calibration complete: {result}")
            elif not self._abort_flag:
                self.state = "done"
                await self.bus.publish(
                    "calibration.result",
                    {
                        "status": "error",
                        "message": f"Need at least 4 points, only got {len(self.cam_points)}",
                    },
                )
            else:
                self.state = "done"
                await self.bus.publish(
                    "calibration.result",
                    {"status": "aborted", "points_collected": len(self.cam_points)},
                )
        except Exception:
            logger.exception("ArUco calibration failed")
            self.state = "done"
            await self.bus.publish(
                "calibration.result",
                {"status": "error", "message": "Calibration failed unexpectedly"},
            )
        finally:
            self.hw.frame_annotator = None
            if _session is self:
                _session = None

    def skip(self):
        """Signal the detection loop to skip the current position."""
        self._skip_flag = True

    def abort(self):
        """Signal the calibration loop to stop."""
        self._abort_flag = True

    async def wait(self):
        """Wait for the background task to complete."""
        if self._task is not None:
            await self._task


class MultiArmArucoCalibrationSession:
    """Calibrates ARM1, ARM2, ARM3 sequentially with fixed marker assignments.

    ARM1 → marker 0, ARM2 → marker 1, ARM3 → marker 2.
    Each arm goes through all CALIBRATION_POSITIONS, detecting its assigned marker.
    """

    mode = "aruco_all"
    SETTLE_FRAMES = 5
    MAX_ATTEMPTS = 150

    def __init__(self, hw: HardwareManager, bus: EventBus):
        from agent_v2.calibration import CALIBRATION_POSITIONS

        self.hw = hw
        self.bus = bus
        self.positions = CALIBRATION_POSITIONS
        self.current_step = 0
        self.state = "idle"

        self._skip_flag = False
        self._abort_flag = False
        self._task: asyncio.Task | None = None

        self._detector = _create_aruco_detector()

        # Per-arm results
        self.arm_results: dict[str, dict] = {}

        # Which arms to calibrate (filtered in start())
        self._target_arms: list[str] = []
        self._current_arm_index: int = 0

        # Annotator state
        self._ann_arm_label: str = ""
        self._ann_target_marker_id: int = 0
        self._ann_step: int = 0
        self._ann_total: int = 0
        self._ann_settle_count: int = 0
        self._ann_status: str = "idle"

    @property
    def total_steps(self) -> int:
        return len(self.positions)

    def _annotate_frame(self, color_image: np.ndarray) -> None:
        """Frame annotator — draws rich debug overlay for multi-arm calibration."""
        _draw_aruco_debug_overlay(
            color_image,
            self._detector,
            target_marker_id=self._ann_target_marker_id,
            arm_label=self._ann_arm_label,
            step=self._ann_step,
            total_steps=self._ann_total or self.total_steps,
            settle_count=self._ann_settle_count,
            settle_target=self.SETTLE_FRAMES,
            status=self._ann_status,
        )

    async def start(self):
        """Begin multi-arm calibration background task."""
        # Determine which arms to calibrate
        connected = set(self.hw.connected_devices)
        self._target_arms = [d for d in ARM_MARKER_MAP if d in connected]

        logger.info(
            f"Multi-arm start: connected={list(connected)}, "
            f"target_arms={self._target_arms}"
        )

        if not self._target_arms:
            logger.error(
                f"No target arms found. Connected: {list(connected)}, "
                f"expected any of: {list(ARM_MARKER_MAP.keys())}"
            )
            raise HTTPException(
                400, "No target arms (ARM1-ARM3) connected — connect them first"
            )

        # Verify all target arms have ground probed
        unprobed = []
        for device in self._target_arms:
            motion = self.hw.get_motion(device)
            if motion is None or motion.ground_z is None:
                unprobed.append(device)
        if unprobed:
            logger.error(f"Ground not probed on: {unprobed}")
            raise HTTPException(
                400,
                f"Ground not probed on: {', '.join(unprobed)} — probe all arms first",
            )

        self.state = "running"
        self.hw.frame_annotator = self._annotate_frame

        logger.info(
            f"Starting multi-arm calibration for {len(self._target_arms)} arms: "
            f"{', '.join(self._target_arms)}"
        )

        self._task = asyncio.create_task(self._run_loop())

    async def _run_loop(self):
        """Main loop: calibrate each arm sequentially."""
        global _session
        try:
            for arm_idx, device in enumerate(self._target_arms):
                if self._abort_flag:
                    break

                marker_id = ARM_MARKER_MAP[device]
                arm_name = Path(device).name
                self._current_arm_index = arm_idx
                self._ann_arm_label = (
                    f"{arm_name} ({arm_idx + 1}/{len(self._target_arms)})"
                )
                self._ann_target_marker_id = marker_id

                logger.info(
                    f"Starting calibration for {arm_name} (marker_id={marker_id}), "
                    f"{arm_idx + 1}/{len(self._target_arms)}"
                )

                # Switch active arm
                self.hw.active_arm_device = device

                # Home all other connected arms
                await self._home_other_arms(device)

                # Run detection loop for this arm
                result = await self._calibrate_single_arm(device, marker_id, arm_idx)
                self.arm_results[device] = result

                # Publish per-arm result
                await self.bus.publish(
                    "calibration.arm_result",
                    {"device": device, "arm_name": arm_name, **result},
                )

                if result["status"] == "error":
                    logger.warning(
                        f"{arm_name} calibration failed: {result.get('message')}"
                    )
                else:
                    logger.info(
                        f"{arm_name} calibration complete: RMSE={result['rmse_mm']}mm "
                        f"({result['quality']}), {result['points_used']} points"
                    )

            # Final summary
            if not self._abort_flag:
                summary = {
                    "status": "done",
                    "arms": {Path(d).name: r for d, r in self.arm_results.items()},
                    "arms_completed": sum(
                        1 for r in self.arm_results.values() if r["status"] == "done"
                    ),
                    "arms_total": len(self._target_arms),
                }
                self.state = "done"
                await self.bus.publish("calibration.result", summary)
                logger.info(f"Multi-arm calibration complete: {summary}")
            else:
                self.state = "done"
                await self.bus.publish(
                    "calibration.result",
                    {"status": "aborted", "arms_completed": len(self.arm_results)},
                )
        except Exception:
            logger.exception("Multi-arm calibration failed")
            self.state = "done"
            await self.bus.publish(
                "calibration.result",
                {
                    "status": "error",
                    "message": "Multi-arm calibration failed unexpectedly",
                },
            )
        finally:
            self.hw.frame_annotator = None
            if _session is self:
                _session = None

    async def _home_other_arms(self, active_device: str):
        """Move all arms except the active one out of the camera's view."""
        others = [d for d in self.hw.connected_devices if d != active_device]
        if not others:
            return
        logger.info(f"Moving {len(others)} other arm(s) out of the way")
        for device in others:
            motion = self.hw.get_motion(device)
            if motion is not None:
                lock = self.hw.get_arm_lock(device)
                async with lock:
                    # Retract arm close to base, high up and out of the way
                    await self.hw.run_in_hw_thread(motion.move_to, 150, 0, 200)
                    await asyncio.sleep(1.5)

    async def _calibrate_single_arm(
        self, device: str, marker_id: int, arm_idx: int
    ) -> dict:
        """Run the detection loop for a single arm across all positions."""
        motion = self.hw.get_motion(device)
        if motion is None:
            return {"status": "error", "message": f"No motion for {device}"}

        lock = self.hw.get_arm_lock(device)
        cam_points: list[np.ndarray] = []
        arm_points: list[np.ndarray] = []

        for i, (x, y, z) in enumerate(self.positions):
            if self._abort_flag:
                break

            self.current_step = i
            self.state = "moving"
            self._ann_step = i + 1
            self._ann_total = self.total_steps
            self._ann_status = "moving"
            self._ann_settle_count = 0

            logger.info(
                f"Position {i + 1}/{self.total_steps}: moving to ({x}, {y}, {z})"
            )

            await self.bus.publish(
                "calibration.progress",
                {
                    "step": i + 1,
                    "total_steps": self.total_steps,
                    "status": "moving",
                    "point_count": len(cam_points),
                    "device": device,
                    "arm_index": arm_idx,
                    "arm_total": len(self._target_arms),
                },
            )

            # Move arm
            async with lock:
                await self.hw.run_in_hw_thread(motion.move_to, x, y, z)
            await asyncio.sleep(1.0)

            if self._abort_flag:
                break

            # Get actual arm position
            async with lock:
                ax, ay, az = await self.hw.run_in_hw_thread(motion.get_pose)
            arm_xyz = np.array([ax, ay, az])

            self.state = "detecting"
            self._ann_status = "detecting"

            # Detection loop
            collected_centers: list[np.ndarray] = []
            attempt = 0
            self._skip_flag = False

            while attempt < self.MAX_ATTEMPTS:
                if self._abort_flag or self._skip_flag:
                    break

                bundle = self.hw.latest_frame
                if bundle is None:
                    await asyncio.sleep(1 / 30)
                    attempt += 1
                    continue

                corners, ids, _ = self._detector.detectMarkers(bundle.raw_color_image)
                result = _find_target_marker(corners, ids, marker_id)

                if result is not None:
                    _, center = result
                    collected_centers.append(center)
                    self._ann_settle_count = len(collected_centers)
                    logger.info(
                        f"Position {i + 1}/{self.total_steps}: detected marker "
                        f"{marker_id} at pixel ({int(center[0])}, {int(center[1])}), "
                        f"settle {len(collected_centers)}/{self.SETTLE_FRAMES}"
                    )

                if attempt % 10 == 0 or len(collected_centers) == self.SETTLE_FRAMES:
                    await self.bus.publish(
                        "calibration.progress",
                        {
                            "step": i + 1,
                            "total_steps": self.total_steps,
                            "status": "detecting",
                            "point_count": len(cam_points),
                            "settle_count": len(collected_centers),
                            "settle_target": self.SETTLE_FRAMES,
                            "position": {
                                "x": round(ax, 1),
                                "y": round(ay, 1),
                                "z": round(az, 1),
                            },
                            "device": device,
                            "arm_index": arm_idx,
                            "arm_total": len(self._target_arms),
                        },
                    )

                if len(collected_centers) >= self.SETTLE_FRAMES:
                    break

                await asyncio.sleep(1 / 30)
                attempt += 1

            if self._abort_flag:
                break

            if self._skip_flag:
                logger.info(f"Position {i + 1} skipped by user")
                continue

            if len(collected_centers) < self.SETTLE_FRAMES:
                logger.warning(
                    f"Position {i + 1}: no marker detected after {attempt} frames, skipping"
                )
                await self.bus.publish(
                    "calibration.progress",
                    {
                        "step": i + 1,
                        "total_steps": self.total_steps,
                        "status": "timeout",
                        "point_count": len(cam_points),
                        "device": device,
                        "arm_index": arm_idx,
                        "arm_total": len(self._target_arms),
                    },
                )
                continue

            # Average marker centers and deproject
            avg_center = np.mean(collected_centers, axis=0)
            px, py = int(round(avg_center[0])), int(round(avg_center[1]))

            cam_3d = self.hw.ct.deproject_patch(
                px, py, bundle.depth_frame, patch_size=15
            )
            if cam_3d is None:
                logger.warning(
                    f"Position {i + 1}: no valid depth at ({px}, {py}), skipping"
                )
                continue

            cam_points.append(cam_3d)
            arm_points.append(arm_xyz)
            logger.info(
                f"Position {i + 1}: averaged center ({px},{py}), "
                f"cam_3d={cam_3d}, arm={arm_xyz}"
            )

        # Solve if enough points
        if len(cam_points) >= 4:
            return _solve_and_save(self.hw, cam_points, arm_points, device=device)
        else:
            return {
                "status": "error",
                "message": f"Need at least 4 points, only got {len(cam_points)}",
            }

    def skip(self):
        """Signal the detection loop to skip the current position."""
        self._skip_flag = True

    def abort(self):
        """Signal the calibration loop to stop."""
        self._abort_flag = True

    async def wait(self):
        """Wait for the background task to complete."""
        if self._task is not None:
            await self._task


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
        mode=_session.mode if _session else "manual",
    )


@router.post("/start")
async def start_calibration(body: StartCalibrationRequest, request: Request):
    hw: HardwareManager = request.app.state.hardware
    bus: EventBus = request.app.state.event_bus
    global _session

    logger.info(f"Calibration start requested: mode={body.mode}, marker_id={body.marker_id}")
    logger.info(
        f"Hardware state: ct={'yes' if hw.ct else 'no'}, "
        f"motion={'yes' if hw.motion else 'no'}, "
        f"active_arm={hw.active_arm_device}, "
        f"connected={hw.connected_devices}"
    )

    if hw.ct is None:
        logger.error("Calibration rejected: hw.ct is None (hardware not ready)")
        raise HTTPException(503, "Hardware not ready")

    if body.mode == "aruco_all":
        # Multi-arm mode — validates arms inside start()
        _session = MultiArmArucoCalibrationSession(hw, bus)
        await _session.start()
        return {
            "status": "started",
            "mode": "aruco_all",
            "total_steps": _session.total_steps,
            "arms": len(_session._target_arms),
        }

    # Single-arm modes require active arm + ground probe
    if hw.motion is None:
        logger.error("Calibration rejected: no arm connected")
        raise HTTPException(503, "Hardware not ready — no arm connected")

    if hw.motion.ground_z is None:
        logger.error("Calibration rejected: ground not probed on active arm")
        raise HTTPException(400, "Ground not probed — probe ground before calibrating")

    if body.mode == "aruco":
        _session = ArucoCalibrationSession(hw, bus, marker_id=body.marker_id)
        await _session.start()
        return {
            "status": "started",
            "mode": "aruco",
            "total_steps": _session.total_steps,
        }

    # Default: manual mode
    _session = WebCalibrationSession(hw, bus)
    active_device = hw.active_arm_device
    if active_device is None:
        raise HTTPException(503, "No active arm device")
    async with hw.get_arm_lock(active_device):
        await _session.start()
    return {
        "status": "started",
        "mode": "manual",
        "total_steps": _session.total_steps,
    }


@router.post("/click")
async def record_click(body: ClickRequest, request: Request):
    hw: HardwareManager = request.app.state.hardware
    global _session

    if _session is None:
        raise HTTPException(400, "No calibration session active")

    if isinstance(_session, (ArucoCalibrationSession, MultiArmArucoCalibrationSession)):
        raise HTTPException(400, "Cannot click in ArUco mode — detection is automatic")

    active_device = hw.active_arm_device
    if active_device is None:
        raise HTTPException(503, "No active arm device")
    async with hw.get_arm_lock(active_device):
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

    if isinstance(_session, (ArucoCalibrationSession, MultiArmArucoCalibrationSession)):
        _session.skip()
        return {"status": "skip_requested"}

    active_device = hw.active_arm_device
    if active_device is None:
        raise HTTPException(503, "No active arm device")
    async with hw.get_arm_lock(active_device):
        result = await _session.skip()

    if result.get("status") == "done":
        _session = None

    return result


@router.post("/abort")
async def abort_calibration(request: Request):
    global _session

    if _session is None:
        raise HTTPException(400, "No calibration session active")

    if isinstance(_session, (ArucoCalibrationSession, MultiArmArucoCalibrationSession)):
        _session.abort()
        # Wait briefly for cleanup, but don't block forever
        try:
            await asyncio.wait_for(_session.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Calibration session did not stop within timeout")
        _session = None
        return {"status": "aborted"}

    points = len(_session.cam_points)
    _session = None
    return {"status": "aborted", "points_collected": points}
