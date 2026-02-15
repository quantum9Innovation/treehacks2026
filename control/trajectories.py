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


def main():
    # ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"]
    ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"]
    # ports = [detect_serial_port()]

    if len(ports) == 0:
        print("Error: No USB serial device found. Connect the arm or use --port.")
        sys.exit(1)

    print(f"Connecting to RoArm-M2 on {ports}...")
    arms = [roarm(roarm_type="roarm_m2", port=port, baudrate=115200) for port in ports]
    for arm in arms:
        arm.echo_set(0)
        arm.torque_set(1)

    print("Moving to home position...")

    for arm in arms:
        arm.move_init()
        arm.pose_ctrl([250, 0, 200, 0])
        time.sleep(1)

    arms[0]
    standby = arms[1:]
    trajectory = "trajectories/lemniscate.json"
    try:
        while True:
            load(trajectory)["trajectory"]
            load(trajectory)["delay"]
            for arm in standby:
                arm.pose_ctrl([250, 0, 200, 0])
            # for x, y, z, t in stream:
            #     action_arm.pose_ctrl([x, y, z, t])
            #     time.sleep(local_delay)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
