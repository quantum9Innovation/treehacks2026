import glob
import sys
import time
import math
from typing import Callable, NewType

from read import load
from roarm_sdk.roarm import roarm

SERIAL_GLOB_PATTERNS = [
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
    "/dev/ARM*",
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


def quad_spring(monitor_arm, right_arm, back_arm, left_arm):
    data = load("trajectories/spring.json")
    data2 = load("trajectories/springShift1.json")
    data3 = load("trajectories/springShift2.json")
    data4 = load("trajectories/springShift3.json")
    stream = data["trajectory"]
    stream2 = data2["trajectory"]
    stream3 = data3["trajectory"]
    stream4 = data4["trajectory"]
    n = min(len(stream), len(stream2), len(stream3), len(stream4))
    delay = data["delay"]

    for i in range(n):
        x1, y1, z1, t1 = stream[i]
        x2, y2, z2, t2 = stream2[i]
        x3, y3, z3, t3 = stream3[i]
        x4, y4, z4, t4 = stream4[i]
        monitor_arm.pose_ctrl([x1, y1, z1, t1])
        back_arm.pose_ctrl([x2, y2, z2, t2])
        right_arm.pose_ctrl([x3, y3, z3, t3])
        left_arm.pose_ctrl([x4, y4, z4, t4])
        time.sleep(delay)


def six_seven(monitor_arm, right_arm, back_arm, left_arm):
    right_arm.pose_ctrl([100, 0, 75, 0])
    left_arm.pose_ctrl([100, 0, 75, 0])

    min_theta = 90
    max_theta = 160
    theta_diff = max_theta - min_theta
    theta_baseline = (min_theta + max_theta) / 2

    n = 10
    k = 2 * math.pi / n

    for t in range(n):
        monitor_arm.joints_angle_ctrl(
            [0, -15, theta_baseline + theta_diff / 2 * math.sin(t * k), 0], 1024, 128
        )
        back_arm.joints_angle_ctrl(
            [
                0,
                -15,
                theta_baseline + theta_diff / 2 * math.sin(t * k + math.pi / 2),
                0,
            ],
            1024,
            128,
        )
        left_arm.joints_angle_ctrl(
            [0, -15, theta_baseline + theta_diff / 2 * math.sin(t * k), 0], 1024, 128
        )
        right_arm.joints_angle_ctrl(
            [
                0,
                -15,
                theta_baseline + theta_diff / 2 * math.sin(t * k + math.pi / 2),
                0,
            ],
            1024,
            128,
        )
        time.sleep(0.05)


def helix(monitor_arm, right_arm, back_arm, left_arm):
    data = load("trajectories/helix.json")
    stream = data["trajectory"]
    delay = data["delay"]

    for x, y, z, t in stream:
        right_arm.pose_ctrl([x, y, z, t])
        left_arm.pose_ctrl([x, y, z, t])
        monitor_arm.pose_ctrl([x, y, z, t])
        back_arm.pose_ctrl([100, 0, 75, 0])
        time.sleep(delay)


def lorenz(monitor_arm, right_arm, back_arm, left_arm):
    data = load("trajectories/lorenz.json")
    stream = data["trajectory"]
    delay = data["delay"]

    for x, y, z, t in stream:
        right_arm.pose_ctrl([x, y, z, t])
        left_arm.pose_ctrl([x, y, z, t])
        monitor_arm.pose_ctrl([x, y, z, t])
        back_arm.pose_ctrl([x, y, z, t])
        time.sleep(delay)


def rossler(monitor_arm, right_arm, back_arm, left_arm):
    data = load("trajectories/rossler.json")
    stream = data["trajectory"]
    delay = data["delay"]

    for x, y, z, t in stream:
        right_arm.pose_ctrl([x, y, z, t])
        left_arm.pose_ctrl([x, y, z, t])
        monitor_arm.pose_ctrl([x, y, z, t])
        back_arm.pose_ctrl([x, y, z, t])
        time.sleep(delay)


def main():
    back_port = "/dev/ttyUSB0"
    right_port = "/dev/ttyUSB1"
    monitor_port = "/dev/ttyUSB2"
    left_port = "/dev/ttyUSB3"
    ports = [right_port, back_port, monitor_port, left_port]

    if len(ports) == 0:
        print("Error: No USB serial device found. Connect the arm or use --port.")
        sys.exit(1)

    print(f"Connecting to RoArm-M2 on {ports}...")
    right_arm = roarm(roarm_type="roarm_m2", port=right_port, baudrate=115200)
    right_arm.echo_set(0)
    right_arm.torque_set(1)

    back_arm = roarm(roarm_type="roarm_m2", port=back_port, baudrate=115200)
    back_arm.echo_set(0)
    back_arm.torque_set(1)

    monitor_arm = roarm(roarm_type="roarm_m2", port=monitor_port, baudrate=115200)
    monitor_arm.echo_set(0)
    monitor_arm.torque_set(1)

    left_arm = roarm(roarm_type="roarm_m2", port=left_port, baudrate=115200)
    left_arm.echo_set(0)
    left_arm.torque_set(1)

    try:
        while True:
            double_lemniscate(monitor_arm, right_arm, back_arm, left_arm)
            quad_spring(monitor_arm, right_arm, back_arm, left_arm)
            six_seven(monitor_arm, right_arm, back_arm, left_arm)
            helix(monitor_arm, right_arm, back_arm, left_arm)
            lorenz(monitor_arm, right_arm, back_arm, left_arm)
            rossler(monitor_arm, right_arm, back_arm, left_arm)

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
