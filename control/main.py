import curses
import glob
import math
import sys
import time
from typing import Callable, NewType

from roarm_sdk.roarm import roarm

from read import load

SERIAL_GLOB_PATTERNS = [
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
]

speed: float = 25
turning: float = 10
skip: bool = False

Pose = NewType("Pose", tuple[float, float, float, float])
clip: Callable[[float, float, float], float] = lambda x, a, b: max(min(x, b), a)


def move_up(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x, y, z + speed, t))


def move_down(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x, y, z - speed, t))


def move_left(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x, y + speed, z, t))


def move_right(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x, y - speed, z, t))


def move_forward(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x + speed, y, z, t))


def move_backward(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x - speed, y, z, t))


def open(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x, y, z, t + turning))


def close(pose: Pose) -> Pose:
    x, y, z, t = pose
    return Pose((x, y, z, t - turning))


def correct(pose: Pose) -> Pose:
    x, y, z, t = pose
    x = clip(x, -490, 490)
    y = clip(y, -490, 490)
    z = clip(z, 0, 490)
    t = clip(t, 0, 90)

    if math.hypot(x, y, z) > 490:
        x = 250
        y = 0
        z = 250

    return Pose((x, y, z, t))


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


def main(stdscr):
    stdscr.nodelay(True)
    port = detect_serial_port()

    if port is None:
        print("Error: No USB serial device found. Connect the arm or use --port.")
        sys.exit(1)

    print(f"Connecting to RoArm-M2 on {port}...")
    arm = roarm(roarm_type="roarm_m2", port=port, baudrate=115200)
    arm.echo_set(0)
    arm.torque_set(1)

    print("Moving to home position...")
    arm.move_init()
    time.sleep(1)

    pose = Pose((250, 0, 250, 0))
    delay: float = 0.05

    try:
        if not skip:
            stream = load("data.json")["trajectory"]
            for x, y, z, t in stream:
                pose = Pose((x, y, z, t))
                arm.pose_ctrl([x, y, z, t])
                time.sleep(delay)

        pose = Pose((250, 0, 250, 0))
        while True:
            x, y, z, t = pose
            arm.pose_ctrl([x, y, z, t])
            time.sleep(delay)

            try:
                key = stdscr.getkey()
            except curses.error:
                continue

            if key == "w":
                pose = move_up(pose)
            if key == "a":
                pose = move_left(pose)
            if key == "s":
                pose = move_down(pose)
            if key == "d":
                pose = move_right(pose)
            if key == "q":
                pose = move_backward(pose)
            if key == "e":
                pose = move_forward(pose)
            if key == "r":
                pose = open(pose)
            if key == "f":
                pose = close(pose)

            pose = correct(pose)

    except KeyboardInterrupt:
        arm.move_init()
        print("Keyboard interrupt")


curses.wrapper(main)
