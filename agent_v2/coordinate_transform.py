"""Coordinate transform: pixel+depth → camera 3D → arm 3D."""

import logging
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from .calibration import DEFAULT_CALIBRATION_PATH, load_calibration
from .models import DetectedObject, Point3D, SceneState

logger = logging.getLogger("agent_v2.coordinate_transform")


class CoordinateTransform:
    """Transforms pixel coordinates to arm coordinates via depth + calibration."""

    def __init__(self, calibration_path: Path = DEFAULT_CALIBRATION_PATH):
        self._intrinsics: rs.intrinsics | None = None
        self._depth_scale: float = 0.001  # default, updated from device
        self._align: rs.align = rs.align(rs.stream.color)

        # Load calibration if available
        self._R: np.ndarray | None = None
        self._t: np.ndarray | None = None
        if calibration_path.exists():
            try:
                print(f"Loading calibration from: {calibration_path}")
                self._R, self._t = load_calibration(calibration_path)
            except Exception as e:
                print(
                    f"WARNING: Failed to load calibration from {calibration_path}: {e}\n"
                    "  Arm coordinates will NOT be available.\n"
                    "  Objects will have depth and camera-frame 3D but no arm positions.\n"
                    "  The goto() command will fail.\n"
                    "  To fix: run 'uv run vlm-agent-v2 --calibrate'"
                )
        else:
            print(
                f"WARNING: No calibration file found at:\n"
                f"  {calibration_path}\n"
                "\n"
                "  Without calibration, the agent cannot convert camera coordinates\n"
                "  to arm coordinates. Object detection will still work, but the\n"
                "  goto() command will fail because arm positions are unknown.\n"
                "\n"
                "  To calibrate, run:\n"
                "    uv run vlm-agent-v2 --calibrate\n"
                "\n"
                "  Or type 'calibrate' in the interactive REPL.\n"
            )

    def set_intrinsics_from_profile(self, profile: rs.pipeline_profile) -> None:
        """Extract and store camera intrinsics from a running pipeline profile.

        We use COLOR stream intrinsics because depth frames are aligned to
        the color frame via rs.align(rs.stream.color). After alignment the
        depth pixels live in the color camera's coordinate system, so
        deprojection must use color focal length / principal point.

        Call this once after camera.start().
        """
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = color_stream.get_intrinsics()
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        print(
            f"Camera intrinsics (color): {self._intrinsics.width}x{self._intrinsics.height}, "
            f"fx={self._intrinsics.fx:.1f}, fy={self._intrinsics.fy:.1f}, "
            f"ppx={self._intrinsics.ppx:.1f}, ppy={self._intrinsics.ppy:.1f}, "
            f"depth_scale={self._depth_scale}"
        )

    def set_intrinsics_from_camera(self, camera) -> None:
        """Extract intrinsics from a RealSenseCamera instance."""
        self.set_intrinsics_from_profile(camera.pipeline.get_active_profile())

    @property
    def has_calibration(self) -> bool:
        return self._R is not None and self._t is not None

    def get_aligned_frames(self, frames: rs.composite_frame) -> rs.composite_frame:
        """Align depth to color frame."""
        return self._align.process(frames)

    def get_depth_at_pixel(
        self, depth_frame: rs.depth_frame, u: int, v: int, patch_size: int = 5
    ) -> float | None:
        """Get depth at pixel (u, v) using median over a patch for noise robustness.

        Args:
            depth_frame: RealSense depth frame
            u: X pixel coordinate
            v: Y pixel coordinate
            patch_size: Size of patch for median filtering

        Returns:
            Depth in millimeters, or None if no valid depth
        """
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        half = patch_size // 2

        depths = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px, py = u + dx, v + dy
                if 0 <= px < w and 0 <= py < h:
                    d = depth_frame.get_distance(px, py)
                    if d > 0:
                        depths.append(d)

        if not depths:
            return None

        depth_m = float(np.median(depths))
        return depth_m * 1000.0  # convert to mm

    def deproject_pixel(
        self, u: int, v: int, depth_frame: rs.depth_frame | None = None, depth_mm: float | None = None
    ) -> np.ndarray | None:
        """Deproject a pixel + depth to camera-frame 3D point.

        Args:
            u: X pixel coordinate
            v: Y pixel coordinate
            depth_frame: Depth frame (used to get depth if depth_mm not provided)
            depth_mm: Depth in mm (if already known)

        Returns:
            (3,) array [x, y, z] in mm in camera frame, or None if no depth
        """
        if self._intrinsics is None:
            logger.error(
                "Cannot deproject: camera intrinsics not set. "
                "Call set_intrinsics_from_camera() after starting the camera."
            )
            return None

        if depth_mm is None:
            if depth_frame is None:
                logger.error(
                    "Cannot deproject pixel (%d, %d): no depth_frame provided and "
                    "no depth_mm given. Pass a depth_frame from aligned capture.", u, v
                )
                return None
            depth_mm = self.get_depth_at_pixel(depth_frame, u, v)
            if depth_mm is None:
                logger.warning(
                    "Cannot deproject pixel (%d, %d): all depth values in 5x5 patch "
                    "are zero (object too close, too far, or in a shadow).", u, v
                )
                return None

        depth_m = depth_mm / 1000.0
        point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [float(u), float(v)], depth_m)
        # Convert to mm
        return np.array([point[0] * 1000, point[1] * 1000, point[2] * 1000])

    def camera_to_arm(self, cam_point: np.ndarray) -> np.ndarray | None:
        """Transform camera-frame 3D point to arm-frame 3D point.

        Args:
            cam_point: (3,) array in camera frame (mm)

        Returns:
            (3,) array in arm frame (mm), or None if not calibrated
        """
        if self._R is None or self._t is None:
            return None
        return self._R @ cam_point + self._t

    def enrich_detections(
        self,
        detections: list[DetectedObject],
        depth_frame: rs.depth_frame,
    ) -> None:
        """Enrich detected objects with depth and 3D coordinates.

        Fills depth_mm, position_cam, and position_arm on each object in-place.
        """
        for obj in detections:
            u, v = obj.center_px

            # Get depth
            depth_mm = self.get_depth_at_pixel(depth_frame, u, v)
            if depth_mm is None:
                logger.debug(f"No depth for object {obj.object_id} ({obj.label}) at ({u},{v})")
                continue
            obj.depth_mm = depth_mm

            # Deproject to camera 3D
            cam_3d = self.deproject_pixel(u, v, depth_mm=depth_mm)
            if cam_3d is not None:
                obj.position_cam = Point3D.from_array(cam_3d)

                # Transform to arm coordinates
                arm_3d = self.camera_to_arm(cam_3d)
                if arm_3d is not None:
                    obj.position_arm = Point3D.from_array(arm_3d)

    def enrich_scene(
        self,
        scene: SceneState,
        depth_frame: rs.depth_frame,
    ) -> None:
        """Enrich a full scene with depth and 3D coordinates."""
        self.enrich_detections(scene.objects, depth_frame)
