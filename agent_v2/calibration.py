"""Camera-to-arm calibration via point correspondence and SVD solver."""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger("agent_v2.calibration")

DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "calibration_data.json"


def solve_rigid_transform(
    cam_points: np.ndarray,
    arm_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve for rigid transform (R, t) mapping camera 3D → arm 3D.

    Uses SVD-based Procrustes alignment (least squares optimal).

    Args:
        cam_points: (N, 3) array of camera-frame 3D points
        arm_points: (N, 3) array of corresponding arm-frame 3D points

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        rmse: Root mean square error in mm

    Raises:
        ValueError: If fewer than 4 point pairs are provided
    """
    if len(cam_points) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(cam_points)}")

    # Compute centroids
    centroid_cam = cam_points.mean(axis=0)
    centroid_arm = arm_points.mean(axis=0)

    # Center both point clouds
    centered_cam = cam_points - centroid_cam
    centered_arm = arm_points - centroid_arm

    # Compute cross-covariance matrix
    H = centered_cam.T @ centered_arm

    # SVD
    U, _S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Handle reflection case
    d = np.linalg.det(V @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])

    # Optimal rotation
    R = V @ sign_matrix @ U.T

    # Translation
    t = centroid_arm - R @ centroid_cam

    # Compute RMSE
    transformed = (R @ cam_points.T).T + t
    errors = np.linalg.norm(transformed - arm_points, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    return R, t, rmse


def save_calibration(
    R: np.ndarray,
    t: np.ndarray,
    rmse: float,
    path: Path = DEFAULT_CALIBRATION_PATH,
) -> None:
    """Save calibration to JSON."""
    data = {
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "rmse_mm": rmse,
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Calibration saved to: {path}")
    logger.info(f"Calibration saved to {path} (RMSE: {rmse:.2f}mm)")


def load_calibration(
    path: Path = DEFAULT_CALIBRATION_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Load calibration from JSON.

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Raises:
        FileNotFoundError: If calibration file doesn't exist
    """
    data = json.loads(path.read_text())
    R = np.array(data["rotation"])
    t = np.array(data["translation"])
    logger.info(f"Calibration loaded from {path} (RMSE: {data['rmse_mm']:.2f}mm)")
    return R, t


class CalibrationProcedure:
    """Interactive calibration using point correspondence.

    Procedure:
    1. Move arm to predefined positions spanning the workspace
    2. At each position, show camera view in OpenCV window
    3. User clicks on arm tip/gripper in the image
    4. Record (camera_3d, arm_coords) pairs
    5. Solve for rigid transform via SVD
    """

    # Predefined arm positions spanning the workspace (x, y, z, t).
    # Wide spread across all three axes is critical for the SVD solver,
    # especially Z which is flipped on inverted-mount arms.
    CALIBRATION_POSITIONS = [
        # Low Z positions
        (250, 0, 25, 0),
        (250, 100, 25, 0),
        (250, -100, 25, 0),
        # Mid Z positions
        (200, 0, 130, 0),
        (200, 100, 130, 0),
        (200, -100, 130, 0),
        # High Z positions
        (150, 0, 240, 0),
        (150, 50, 240, 0),
        (150, -50, 240, 0),
        # Extra spread
        (300, 0, 75, 0),
    ]

    def __init__(self, camera, arm, coordinate_transform):
        """
        Args:
            camera: RealSenseCamera instance (started)
            arm: RobotArmController instance
            coordinate_transform: CoordinateTransform instance (for deprojection)
        """
        self.camera = camera
        self.arm = arm
        self.ct = coordinate_transform
        self._clicked_pixel: tuple[int, int] | None = None
        self._color_width: int = 0  # set during run, used to remap depth-side clicks

    def _mouse_callback(self, event, x, y, _flags, _param):
        """OpenCV mouse callback — clicks on either half map to the same pixel."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # If click is on the depth (right) half, remap to color coordinates
            if self._color_width > 0 and x >= self._color_width:
                x = x - self._color_width
            self._clicked_pixel = (x, y)

    def _capture_raw_aligned(self) -> tuple[np.ndarray, np.ndarray, rs.depth_frame]:
        """Capture aligned color + depth without any overlays.

        Returns:
            color_image: Raw BGR numpy array (no coordinate axes)
            depth_colorized: Colorized depth numpy array
            depth_frame: Raw depth frame for coordinate queries
        """
        frames = self.camera.pipeline.wait_for_frames()
        aligned = self.ct.get_aligned_frames(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")

        color_image = np.asanyarray(color_frame.get_data())

        colorized = self.camera.colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized.get_data())

        return color_image, depth_image, depth_frame

    def run(self, save_path: Path = DEFAULT_CALIBRATION_PATH) -> tuple[np.ndarray, np.ndarray, float]:
        """Run the interactive calibration procedure.

        Returns:
            R, t, rmse from the solved transform
        """
        cam_points = []
        arm_points = []

        window_name = "Calibration - Click on arm tip (left = color, right = depth)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "=" * 60)
        print("CALIBRATION PROCEDURE")
        print("=" * 60)
        print(f"The arm will move to {len(self.CALIBRATION_POSITIONS)} positions.")
        print("At each position, click on the arm tip/gripper in either image.")
        print("Clicks on the depth (right) half are mapped to the same pixel on color.")
        print("Press 's' to skip a position, 'q' to quit early.\n")

        try:
            for i, (x, y, z, t) in enumerate(self.CALIBRATION_POSITIONS):
                print(f"\n--- Position {i + 1}/{len(self.CALIBRATION_POSITIONS)} ---")
                print(f"Moving arm to x={x}, y={y}, z={z}...")

                self.arm.pose_ctrl(x, y, z, t)
                time.sleep(1.0)

                # Get actual arm position (may differ slightly)
                pose = self.arm.pose_get()
                arm_pos = pose["position"]
                arm_xyz = np.array([arm_pos["x"], arm_pos["y"], arm_pos["z"]])

                self._clicked_pixel = None
                # Keep depth_frame from last capture for deprojection on confirm
                current_depth_frame: rs.depth_frame | None = None

                while True:
                    # Capture raw aligned frames (no axis overlays)
                    color_image, depth_image, depth_frame = self._capture_raw_aligned()
                    current_depth_frame = depth_frame
                    self._color_width = color_image.shape[1]

                    # Prepare display copies
                    color_display = color_image.copy()
                    depth_display = depth_image.copy()

                    # Draw instructions on color image
                    cv2.putText(
                        color_display,
                        f"Pos {i + 1}/{len(self.CALIBRATION_POSITIONS)} - Click arm tip",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        color_display,
                        f"Arm: x={arm_pos['x']:.0f} y={arm_pos['y']:.0f} z={arm_pos['z']:.0f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                    if self._clicked_pixel:
                        cx, cy = self._clicked_pixel
                        # Draw crosshair on color
                        cv2.circle(color_display, (cx, cy), 8, (0, 0, 255), 2)
                        cv2.line(color_display, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
                        cv2.line(color_display, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)
                        # Mirror crosshair on depth for visual reference
                        cv2.circle(depth_display, (cx, cy), 8, (0, 0, 255), 2)
                        cv2.line(depth_display, (cx - 12, cy), (cx + 12, cy), (0, 0, 255), 1)
                        cv2.line(depth_display, (cx, cy - 12), (cx, cy + 12), (0, 0, 255), 1)

                        cv2.putText(
                            color_display,
                            "ENTER=confirm  R=retry  S=skip",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            1,
                        )

                    # Label each half
                    cv2.putText(
                        color_display, "COLOR", (10, color_display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )
                    cv2.putText(
                        depth_display, "DEPTH", (10, depth_display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )

                    # Show side by side
                    combined = np.hstack([color_display, depth_display])
                    cv2.imshow(window_name, combined)
                    key = cv2.waitKey(30) & 0xFF

                    if key == ord("q"):
                        print("Calibration aborted by user.")
                        cv2.destroyAllWindows()
                        if len(cam_points) >= 4:
                            print(f"Solving with {len(cam_points)} points collected so far...")
                            break
                        raise RuntimeError("Need at least 4 points for calibration")

                    if key == ord("s"):
                        print("Skipping this position.")
                        break

                    if key == ord("r"):
                        self._clicked_pixel = None
                        continue

                    if key == 13 and self._clicked_pixel:  # Enter
                        px, py = self._clicked_pixel
                        if current_depth_frame is None:
                            print("  ERROR: No depth frame captured yet. Wait for next frame.")
                            self._clicked_pixel = None
                            continue

                        # Check depth at pixel first for a clear error
                        depth_mm = self.ct.get_depth_at_pixel(current_depth_frame, px, py)
                        if depth_mm is None:
                            print(
                                f"  ERROR: No valid depth at pixel ({px}, {py}). "
                                "All depth values in the 5x5 patch are zero. "
                                "The arm tip may be too close/far from the camera, "
                                "or the pixel is in a depth shadow. Try clicking a nearby spot."
                            )
                            self._clicked_pixel = None
                            continue

                        cam_3d = self.ct.deproject_pixel(px, py, depth_frame=current_depth_frame)
                        if cam_3d is None:
                            intrinsics_ok = self.ct._intrinsics is not None
                            print(
                                f"  ERROR: Deprojection failed at pixel ({px}, {py}).\n"
                                f"    Intrinsics loaded: {intrinsics_ok}\n"
                                f"    Depth at pixel: {depth_mm:.1f}mm\n"
                                "    This is unexpected — check camera setup."
                            )
                            self._clicked_pixel = None
                            continue

                        cam_points.append(cam_3d)
                        arm_points.append(arm_xyz)
                        print(
                            f"  Recorded point {len(cam_points)}: "
                            f"pixel=({px},{py}) depth={depth_mm:.0f}mm "
                            f"cam_3d=[{cam_3d[0]:.1f}, {cam_3d[1]:.1f}, {cam_3d[2]:.1f}] "
                            f"arm=[{arm_xyz[0]:.1f}, {arm_xyz[1]:.1f}, {arm_xyz[2]:.1f}]"
                        )
                        break
                else:
                    continue

                if key == ord("q"):
                    break

        finally:
            cv2.destroyAllWindows()

        if len(cam_points) < 4:
            raise RuntimeError(f"Need at least 4 points, only got {len(cam_points)}")

        cam_arr = np.array(cam_points)
        arm_arr = np.array(arm_points)

        R, t_vec, rmse = solve_rigid_transform(cam_arr, arm_arr)

        # Per-point error breakdown
        transformed = (R @ cam_arr.T).T + t_vec
        per_point_errors = np.linalg.norm(transformed - arm_arr, axis=1)

        print(f"\n{'=' * 60}")
        print("CALIBRATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  Points used: {len(cam_points)}")
        print(f"  Overall RMSE: {rmse:.2f}mm")
        if rmse < 10:
            print("  Quality: EXCELLENT")
        elif rmse < 20:
            print("  Quality: Good (usable)")
        else:
            print("  Quality: Poor - consider recalibrating")

        print(f"\n  Per-point errors:")
        for idx, (err, cam_pt, arm_pt, est_pt) in enumerate(
            zip(per_point_errors, cam_arr, arm_arr, transformed)
        ):
            flag = " *** OUTLIER" if err > 2 * rmse and err > 15 else ""
            print(
                f"    Point {idx + 1}: error={err:.1f}mm  "
                f"arm=[{arm_pt[0]:.0f},{arm_pt[1]:.0f},{arm_pt[2]:.0f}] "
                f"estimated=[{est_pt[0]:.0f},{est_pt[1]:.0f},{est_pt[2]:.0f}]{flag}"
            )

        # Show axis mapping for sanity check
        print(f"\n  Rotation matrix (camera → arm axis mapping):")
        axis_labels = ["X", "Y", "Z"]
        for i in range(3):
            components = []
            for j in range(3):
                val = R[i, j]
                if abs(val) > 0.1:
                    sign = "+" if val > 0 else "-"
                    components.append(f"{sign}{abs(val):.2f}*cam_{axis_labels[j]}")
            mapping = " ".join(components) if components else "~0"
            print(f"    arm_{axis_labels[i]} = {mapping}")

        print(f"  Translation: [{t_vec[0]:.1f}, {t_vec[1]:.1f}, {t_vec[2]:.1f}]mm")
        det = np.linalg.det(R)
        print(f"  Rotation det: {det:.4f} (should be +1.0)")
        print()

        save_calibration(R, t_vec, rmse, save_path)
        return R, t_vec, rmse
