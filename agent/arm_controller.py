"""Robot arm controller with Z-coordinate inversion support."""

import glob
import time
from typing import Any

from roarm_sdk.roarm import roarm

SERIAL_GLOB_PATTERNS = [
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
    "/dev/ARM*",
]

DEFAULT_Z_OFFSET = 300.0
DEFAULT_GRIPPER_SPEED = 100
DEFAULT_GRIPPER_ACC = 50


def detect_serial_port() -> str | None:
    """Auto-detect the serial port for the robot arm."""
    devices = []
    for pattern in SERIAL_GLOB_PATTERNS:
        devices.extend(glob.glob(pattern))
    if len(devices) == 1:
        return devices[0]
    if len(devices) > 1:
        print(f"Multiple serial devices found: {devices}")
        print("Please specify one with --port")
    return None


class RobotArmController:
    """Wrapper for RoArm-M2 with Z-coordinate inversion support."""

    def __init__(
        self,
        port: str | None = None,
        z_offset: float = DEFAULT_Z_OFFSET,
        invert_z: bool = True,
    ):
        """
        Initialize the robot arm controller.

        Args:
            port: Serial port (auto-detect if None)
            z_offset: Z offset for coordinate transformation (default: 350mm)
            invert_z: If True, invert Z axis for upside-down mounting
        """
        self.port = port or detect_serial_port()
        if self.port is None:
            raise RuntimeError(
                "No USB serial device found. Connect the arm or use --port."
            )

        self.z_offset = z_offset
        self.invert_z = invert_z

        print(f"Connecting to RoArm-M2 on {self.port}...")
        self.arm = roarm(roarm_type="roarm_m2", port=self.port, baudrate=115200)

    def _transform_z(self, z: float) -> float:
        """Transform Z coordinate from user space to robot space."""
        if self.invert_z:
            return self.z_offset - z
        return z

    def _inverse_transform_z(self, z: float) -> float:
        """Transform Z coordinate from robot space to user space."""
        if self.invert_z:
            return self.z_offset - z
        return z

    # Position limits
    X_MIN, X_MAX = 50.0, 400.0
    Y_MIN, Y_MAX = -400.0, 400.0
    Z_MIN, Z_MAX = 0.0, 300.0
    T_MIN, T_MAX = 0.0, 90.0
    MAX_REACH = 500.0  # Maximum total reach distance from base

    def pose_ctrl(self, x: float, y: float, z: float, t: float) -> dict[str, Any]:
        """
        Move the robot arm to a specific position.

        Args:
            x: X coordinate in mm (forward/backward from base, range: 50-500)
            y: Y coordinate in mm (left/right, range: -500 to 500)
            z: Z coordinate in mm (height, in user coordinates, range: 0-300)
            t: Gripper rotation angle in degrees (range: 0-90)

        Returns:
            Status dict with position info
        """
        import math

        # Validate position limits
        errors = []
        if not (self.X_MIN <= x <= self.X_MAX):
            errors.append(f"X={x} out of range [{self.X_MIN}, {self.X_MAX}]")
        if not (self.Y_MIN <= y <= self.Y_MAX):
            errors.append(f"Y={y} out of range [{self.Y_MIN}, {self.Y_MAX}]")
        if not (self.Z_MIN <= z <= self.Z_MAX):
            errors.append(f"Z={z} out of range [{self.Z_MIN}, {self.Z_MAX}]")
        if not (self.T_MIN <= t <= self.T_MAX):
            errors.append(f"T={t} out of range [{self.T_MIN}, {self.T_MAX}]")

        # Check total reach distance (x, y, z)
        reach = math.sqrt(x * x + y * y + z * z)
        if reach > self.MAX_REACH:
            errors.append(f"Total reach {reach:.1f}mm exceeds max {self.MAX_REACH}mm")

        if errors:
            return {
                "status": "error",
                "message": "Position out of bounds: " + "; ".join(errors),
            }

        transformed_z = self._transform_z(z)
        self.arm.pose_ctrl([x, y, transformed_z, t])
        time.sleep(0.3)
        return {
            "status": "success",
            "position": {"x": x, "y": y, "z": z, "t": t},
            "message": f"Moved to x={x}, y={y}, z={z}, t={t}",
        }

    def pose_get(self) -> dict[str, Any]:
        """
        Get the current position of the robot arm.

        Returns:
            Dict with current x, y, z, t values (z in user coordinates)
        """
        pose = self.arm.pose_get()
        x, y, raw_z, t = pose
        z = self._inverse_transform_z(raw_z)
        return {
            "status": "success",
            "position": {"x": x, "y": y, "z": z, "t": t},
            "message": f"Current position: x={x:.1f}, y={y:.1f}, z={z:.1f}, t={t:.1f}",
        }

    def move_home(self) -> dict[str, Any]:
        """
        Move the robot arm to its home position.

        Returns:
            Status dict
        """
        self.arm.move_init()
        time.sleep(1.5)
        return {"status": "success", "message": "Moved to home position"}

    def gripper_ctrl(self, angle: float) -> dict[str, Any]:
        """
        Control the gripper opening.

        Args:
            angle: Gripper angle (0 = closed, 90 = fully open)

        Returns:
            Status dict
        """
        angle = max(0, min(90, angle))
        self.arm.gripper_angle_ctrl(
            angle=angle, speed=DEFAULT_GRIPPER_SPEED, acc=DEFAULT_GRIPPER_ACC
        )
        time.sleep(0.5)
        state = "open" if angle > 45 else "closed"
        return {
            "status": "success",
            "gripper_angle": angle,
            "message": f"Gripper {state} at {angle} degrees",
        }
