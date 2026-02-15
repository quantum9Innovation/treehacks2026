import glob
import sys
import time
from typing import Callable, NewType

from read import load
from roarm_sdk.roarm import roarm

SERIAL_GLOB_PATTERNS = [
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
]

speed: float = 50
turning: float = 10
skip: bool = False
loop: bool = True

# trajectories: list[str] = sorted(glob.glob("trajectories/*.json"))
trajectories: list[str] = ["trajectories/wave.json"]
Pose = NewType("Pose", tuple[float, float, float, float])
clip: Callable[[float, float, float], float] = lambda x, a, b: max(min(x, b), a)


def detect_serial_port():
    devices = []

    for pattern in SERIAL_GLOB_PATTERNS:
        devices.extend(glob.glob(pattern))

    if len(devices) == 1:
        return devices[0]
    elif len(devices) > 1:
        print(f"Multiple serial devices found: {devices}")
        print("Please specify one with --port")
    else:
        return None


def double_lemniscate(monitor_arm, right_arm, back_arm, left_arm):
    data = load("trajectories/lemniscate.json")
    stream = data["trajectory"]
    delay = data["delay"]

    for x, y, z, t in stream:
        monitor_arm.pose_ctrl([100, 0, 75, 0])
        back_arm.pose_ctrl([100, 0, 75, 0])
        right_arm.pose_ctrl([x, y, z, t])
        left_arm.pose_ctrl([x, y, z, t])

    time.sleep(delay)


def main():
    port_right = "/dev/ttyUSB0"
    back_port = "/dev/ttyUSB1"
    port_monitor = "/dev/ttyUSB2"
    left_port = "/dev/ttyUSB3"
    ports = [port_right, back_port, port_monitor, left_port]

    if len(ports) == 0:
        print("Error: No USB serial device found. Connect the arm or use --port.")
        sys.exit(1)

    print(f"Connecting to RoArm-M2 on {ports}...")
    right_arm = roarm(roarm_type="roarm_m2", port=port_right, baudrate=115200)
    right_arm.echo_set(0)
    right_arm.torque_set(1)

    back_arm = roarm(roarm_type="roarm_m2", port=back_port, baudrate=115200)
    back_arm.echo_set(0)
    back_arm.torque_set(1)

    monitor_arm = roarm(roarm_type="roarm_m2", port=port_monitor, baudrate=115200)
    monitor_arm.echo_set(0)
    monitor_arm.torque_set(1)

    left_arm = roarm(roarm_type="roarm_m2", port=left_port, baudrate=115200)
    left_arm.echo_set(0)
    left_arm.torque_set(1)

    try:
        while True:
            double_lemniscate(monitor_arm, right_arm, back_arm, left_arm)

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
