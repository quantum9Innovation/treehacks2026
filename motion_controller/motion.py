import glob
import math
import os
import sys
import time

from roarm_sdk.roarm import roarm

SERIAL_GLOB_PATTERNS = [
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
]


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

# Arm geometry (calibrated against arm's internal FK over 71 data points, RMS ~8mm)
L1 = 234.63  # shoulder to elbow (mm)
L2 = 285.35  # elbow to end effector (mm)
R_OFF = 22.83  # radial offset from base axis (mm)
Z_OFF = -11.46  # vertical offset (mm)

# Probing
CONTACT_THRESHOLD = 50
PROBE_SPEED = 300
BACKOFF_ANGLE = 0.05
MAX_SHOULDER_ANGLE = math.pi / 2
POLL_INTERVAL = 0.05

POSITION_TOLERANCE = 15.0  # mm (limited by approximate link lengths)
SETTLE_TIMEOUT = 3.0  # seconds


def reachable(x, y, z):
    """Check if (x, y, z) is within the arm's workspace."""
    r = math.sqrt(x * x + y * y) - R_OFF
    z_adj = z - Z_OFF
    d = math.sqrt(r * r + z_adj * z_adj)
    return abs(L1 - L2) < d < L1 + L2


def ik(x, y, z):
    """Inverse kinematics: XYZ (mm) -> joint angles [base, shoulder, elbow, hand]."""
    base = math.atan2(y, x)
    r = math.sqrt(x * x + y * y) - R_OFF
    z_adj = z - Z_OFF

    d_sq = r * r + z_adj * z_adj
    d = math.sqrt(d_sq)

    cos_elbow = (d_sq - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)

    shoulder = math.atan2(r, z_adj) - math.atan2(L2 * math.sin(elbow), L1 + L2 * cos_elbow)

    return [base, shoulder, elbow, 0]


def fk(base, shoulder, elbow):
    """Forward kinematics: joint angles -> XYZ (mm)."""
    r = L1 * math.sin(shoulder) + L2 * math.sin(shoulder + elbow) + R_OFF
    z = L1 * math.cos(shoulder) + L2 * math.cos(shoulder + elbow) + Z_OFF
    x = r * math.cos(base)
    y = r * math.sin(base)
    return x, y, z


class Motion:
    def __init__(self, port: str | None = None, inverted: bool = False):
        resolved_port = port or detect_serial_port()
        if resolved_port is None:
            raise RuntimeError(
                "No USB serial device found. Connect the arm or use --port."
            )
        print(f"Connecting to RoArm-M2 on {resolved_port}...")
        self.arm = roarm(
            roarm_type="roarm_m2",
            port=resolved_port,
            baudrate=115200,
        )
        self.ground_z = None
        self.inverted = inverted
        self.last_move_error_mm = None

    def _to_native_z(self, z):
        """Convert ground-relative Z to native Z."""
        if self.inverted:
            return self.ground_z - z
        return self.ground_z + z

    def _from_native_z(self, native_z):
        """Convert native Z to ground-relative Z."""
        if self.inverted:
            return self.ground_z - native_z
        return native_z - self.ground_z

    def probe_ground(self, base_angle=0.0):
        """Probe ground by moving arm until contact. Stores ground_z.

        Normal mode: sweeps shoulder with shoulder+elbow=π constraint
        (forearm always points straight down).
        Inverted mode: holds shoulder=0 (straight down) and slowly straightens
        elbow from π/2 → 0, extending the arm toward the ground.
        """
        print(f"Probing ground at base={math.degrees(base_angle):.0f}° (inverted={self.inverted})")

        if self.inverted:
            return self._probe_ground_inverted(base_angle)

        # Start: forearm pointing straight down, shoulder slightly tilted out
        start_shoulder = 0.3
        self.arm.joints_radian_ctrl(
            radians=[base_angle, start_shoulder, math.pi - start_shoulder, 0],
            speed=2000,
            acc=200,
        )
        time.sleep(2)

        # Target: shoulder at max, elbow adjusts to keep forearm vertical
        target_elbow = math.pi - MAX_SHOULDER_ANGLE
        self.arm.joints_radian_ctrl(
            radians=[base_angle, MAX_SHOULDER_ANGLE, target_elbow, 0],
            speed=PROBE_SPEED,
            acc=100,
        )

        # Re-baseline after motion starts (skip gravity transient)
        time.sleep(0.5)
        for _ in range(3):
            self._get_torques()
            time.sleep(POLL_INTERVAL)
        baseline_torE = self._get_torques()[2]

        while True:
            time.sleep(POLL_INTERVAL)
            torques = self._get_torques()
            torE_delta = abs(torques[2] - baseline_torE)

            fb = self._feedback()
            shoulder_now = fb[4]
            _, _, actual_z = fk(fb[3], fb[4], fb[5])

            if torE_delta > CONTACT_THRESHOLD:
                shoulder_back = max(shoulder_now - BACKOFF_ANGLE, 0)
                elbow_back = math.pi - shoulder_back
                self.arm.joints_radian_ctrl(
                    radians=[base_angle, shoulder_back, elbow_back, 0],
                    speed=1000,
                    acc=200,
                )
                time.sleep(0.5)
                fb = self._feedback()
                _, _, gz = fk(fb[3], fb[4], fb[5])
                self.ground_z = gz
                print(
                    f"Contact at shoulder={math.degrees(shoulder_now):.1f}° "
                    f"z={actual_z:.0f}mm (torE delta={int(torE_delta)})"
                )
                print(f"Ground Z = {self.ground_z:.0f}mm (native)")
                return self.ground_z

            if shoulder_now >= MAX_SHOULDER_ANGLE - 0.02:
                break

        raise RuntimeError("No ground detected within shoulder range")

    def _probe_ground_inverted(self, base_angle):
        """Inverted probe: shoulder=0 (straight down), slowly straighten elbow.

        Elbow sweeps π/2 → 0, extending the end effector downward.
        """
        shoulder = 0.0
        start_elbow = math.pi / 2
        target_elbow = 0.0

        # Move to start: arm pointing down, elbow bent 90°
        self.arm.joints_radian_ctrl(
            radians=[base_angle, shoulder, start_elbow, 0], speed=2000, acc=200
        )
        time.sleep(2)

        # Start slow elbow sweep, then re-baseline after transient settles
        self.arm.joints_radian_ctrl(
            radians=[base_angle, shoulder, target_elbow, 0],
            speed=PROBE_SPEED,
            acc=100,
        )
        time.sleep(0.5)
        for _ in range(5):
            self._get_torques()
            time.sleep(POLL_INTERVAL)
        baseline = self._get_torques()
        baseline_torS = baseline[1]
        baseline_torE = baseline[2]

        while True:
            time.sleep(POLL_INTERVAL)
            torques = self._get_torques()
            torS_delta = abs(torques[1] - baseline_torS)
            torE_delta = abs(torques[2] - baseline_torE)
            max_delta = max(torS_delta, torE_delta)

            fb = self._feedback()
            elbow_now = fb[5]
            _, _, actual_z = fk(fb[3], fb[4], fb[5])

            if max_delta > CONTACT_THRESHOLD:
                # Back off by bending elbow back up
                elbow_back = min(elbow_now + BACKOFF_ANGLE, start_elbow)
                self.arm.joints_radian_ctrl(
                    radians=[base_angle, shoulder, elbow_back, 0],
                    speed=1000,
                    acc=200,
                )
                time.sleep(0.5)
                fb = self._feedback()
                _, _, gz = fk(fb[3], fb[4], fb[5])
                self.ground_z = gz
                print(
                    f"Contact at elbow={math.degrees(elbow_now):.1f}° "
                    f"z={actual_z:.0f}mm (torS delta={int(torS_delta)}, torE delta={int(torE_delta)})"
                )
                print(f"Ground Z = {self.ground_z:.0f}mm (native)")
                return self.ground_z

            if elbow_now <= target_elbow + 0.02:
                break

        raise RuntimeError("No ground detected within elbow range")

    def move_to(self, x, y, z, speed=2000, acc=200):
        """Move to ground-relative XYZ using IK. z is clamped >= 0."""
        self._require_calibrated()
        z = max(0.0, z)
        native_z = self._to_native_z(z)
        if not reachable(x, y, native_z):
            r = math.sqrt(x * x + y * y)
            print(f"  Unreachable: ({x:.0f}, {y:.0f}, {z:.0f}) r={r:.0f}mm")
            return False
        joints = ik(x, y, native_z)

        # Clamp to joint limits
        joints[1] = max(-math.pi / 2, min(math.pi / 2, joints[1]))
        joints[2] = max(-0.873, min(math.pi, joints[2]))

        self.arm.joints_radian_ctrl(radians=joints, speed=speed, acc=acc)

        # Wait until arm reaches target (use our FK, not arm's internal FK)
        start = time.time()
        err = float("inf")
        while time.time() - start < SETTLE_TIMEOUT:
            time.sleep(0.1)
            fb = self._feedback()
            ax, ay, az = fk(fb[3], fb[4], fb[5])
            err = math.sqrt((x - ax)**2 + (y - ay)**2 + (native_z - az)**2)
            if err < POSITION_TOLERANCE:
                self.last_move_error_mm = err
                return True
        self.last_move_error_mm = err
        print(f"  Settle timeout (error={err:.1f}mm)")
        return False

    def probe_height_at(self, x, y, safe_z=100):
        """Smooth descent at (x,y) from safe_z toward ground, polling torE.
        Returns ground-relative Z of contact (mm), 0 if only ground."""
        self._require_calibrated()

        # Check both safe height and ground level are reachable
        native_safe = self._to_native_z(safe_z)
        if not reachable(x, y, native_safe) or not reachable(x, y, self.ground_z):
            r = math.sqrt(x * x + y * y)
            print(f"  Unreachable: ({x:.0f}, {y:.0f}) r={r:.0f}mm")
            return None

        # Retract to near-vertical (forearm down) so lateral moves stay high
        self.arm.joints_radian_ctrl(
            radians=[math.atan2(y, x), 0.1, math.pi - 0.1, 0], speed=2000, acc=200
        )
        time.sleep(0.5)

        # Move to safe height and settle
        self.move_to(x, y, safe_z)

        # IK for ground level — gives target for both shoulder and elbow
        ground_joints = ik(x, y, self.ground_z)
        ground_joints[1] = max(-math.pi / 2, min(math.pi / 2, ground_joints[1]))
        ground_joints[2] = max(-0.873, min(math.pi, ground_joints[2]))

        # Start smooth descent, then re-baseline after initial transient
        self.arm.joints_radian_ctrl(
            radians=ground_joints, speed=PROBE_SPEED, acc=100
        )
        time.sleep(0.5)
        for _ in range(3):
            self._get_torques()
            time.sleep(POLL_INTERVAL)
        baseline_torE = self._get_torques()[2]

        contact_z = 0.0
        start = time.time()
        while time.time() - start < 10.0:
            time.sleep(POLL_INTERVAL)
            torques = self._get_torques()
            torE_delta = abs(torques[2] - baseline_torE)

            fb = self._feedback()
            _, _, cur_z = fk(fb[3], fb[4], fb[5])

            if torE_delta > CONTACT_THRESHOLD:
                # Stop at current position
                self.arm.joints_radian_ctrl(
                    radians=[fb[3], fb[4], fb[5], fb[6]], speed=1000, acc=200
                )
                time.sleep(0.3)
                contact_z = self._from_native_z(cur_z)
                break

            # Check if we've reached ground level
            if self.inverted:
                if cur_z >= self.ground_z - 10:
                    contact_z = 0.0
                    break
            else:
                if cur_z <= self.ground_z + 10:
                    contact_z = 0.0
                    break

        # Back off to safe height
        safe_joints = ik(x, y, self._to_native_z(safe_z))
        safe_joints[1] = max(-math.pi / 2, min(math.pi / 2, safe_joints[1]))
        safe_joints[2] = max(-0.873, min(math.pi, safe_joints[2]))
        self.arm.joints_radian_ctrl(radians=safe_joints, speed=2000, acc=200)
        time.sleep(0.5)

        return contact_z

    def get_pose(self):
        """Get current ground-relative XYZ (using our FK)."""
        self._require_calibrated()
        fb = self._feedback()
        x, y, z = fk(fb[3], fb[4], fb[5])
        return x, y, self._from_native_z(z)

    def home(self):
        self.arm.joints_radian_ctrl(
            radians=[0, 0, math.pi / 2, 0], speed=2000, acc=200
        )

    def _feedback(self):
        """Call feedback_get() with SDK stdout suppressed."""
        fd = sys.stdout.fileno()
        old = os.dup(fd)
        os.dup2(os.open(os.devnull, os.O_WRONLY), fd)
        try:
            return self.arm.feedback_get()
        finally:
            os.dup2(old, fd)
            os.close(old)

    def _get_torques(self):
        self._feedback()
        d = self.arm.base_controller.base_data
        return d["torB"], d["torS"], d["torE"], d["torH"]

    def _require_calibrated(self):
        if self.ground_z is None:
            raise RuntimeError("Call probe_ground() before using motion commands")


if __name__ == "__main__":
    m = Motion()
    m.probe_ground()

    print(f"\nMoving to Z=150 above ground...")
    m.move_to(0, 200, 150)
    time.sleep(1)
    pose = m.get_pose()
    print(f"Pose: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f}")

    print(f"\nMoving to Z=10 above ground...")
    m.move_to(0, 200, 10)
    time.sleep(1)
    pose = m.get_pose()
    print(f"Pose: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f}")

    print("\nReturning home")
    m.home()
