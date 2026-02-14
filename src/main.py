import argparse
import glob
import math
import sys
import time

from roarm_sdk.roarm import roarm

SERIAL_GLOB_PATTERNS = [
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*"
]
MOVE_SPEED = 600
MOVE_ACC = 100

# Circle in the Y-Z plane (around the X axis)
CIRCLE_X = 250       # fixed X distance from base
CIRCLE_CENTER_Y = 0  # center Y
CIRCLE_CENTER_Z = 200  # center Z
CIRCLE_RADIUS = 80
CIRCLE_POINTS = 72
CIRCLE_T = 0
STEP_DELAY = 0.05


def detect_serial_port():
    devices = []
    for pattern in SERIAL_GLOB_PATTERNS:
        devices.extend(glob.glob(pattern))
    if len(devices) == 1:
        return devices[0]
    if len(devices) > 1:
        print(f"Multiple serial devices found: {devices}")
        print("Please specify one with --port")
    return None


def main():
    parser = argparse.ArgumentParser(description="Make RoArm-M2 draw a circle around the X axis.")
    parser.add_argument("--port", type=str, default=None, help="Serial port (auto-detected if omitted)")
    parser.add_argument("--loops", type=int, default=2, help="Number of circles (default: 2)")
    parser.add_argument("--radius", type=float, default=CIRCLE_RADIUS, help="Circle radius in mm (default: 80)")
    parser.add_argument("--speed", type=int, default=MOVE_SPEED, help="Movement speed, 1-4096 (default: 600)")
    args = parser.parse_args()

    port = args.port or detect_serial_port()
    if port is None:
        print("Error: No USB serial device found. Connect the arm or use --port.")
        sys.exit(1)

    print(f"Connecting to RoArm-M2 on {port}...")
    arm = roarm(roarm_type="roarm_m2", port=port, baudrate=115200)

    print("Moving to home position...")
    arm.move_init()
    time.sleep(2)

    # Move to start of circle (top of circle: Y=0, Z=center+radius)
    start_y = CIRCLE_CENTER_Y
    start_z = CIRCLE_CENTER_Z + args.radius
    print(f"Moving to circle start (x={CIRCLE_X}, y={start_y}, z={start_z:.0f})...")
    arm.pose_ctrl([CIRCLE_X, start_y, start_z, CIRCLE_T])
    time.sleep(1)

    print(f"Drawing {args.loops} circle(s) around X axis, radius={args.radius}mm...")
    try:
        for loop in range(args.loops):
            print(f"  Circle {loop + 1}/{args.loops}")
            for i in range(1, CIRCLE_POINTS + 1):
                angle = 2 * math.pi * i / CIRCLE_POINTS
                y = CIRCLE_CENTER_Y + args.radius * math.sin(angle)
                z = CIRCLE_CENTER_Z + args.radius * math.cos(angle)
                arm.pose_ctrl([CIRCLE_X, y, z, CIRCLE_T])
                time.sleep(STEP_DELAY)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("Returning to home position...")
    arm.move_init()
    time.sleep(2)
    print("Done.")


if __name__ == "__main__":
    main()
 