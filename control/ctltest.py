import argparse
import glob
import sys
import time

from roarm_sdk.roarm import roarm

SERIAL_GLOB_PATTERNS = [
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
    "/dev/ARM*",
]


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


def parse_position(line):
    """Parse position input. Accepts 'x y z' or 'x y z t' format."""
    parts = line.strip().split()
    if len(parts) == 3:
        x, y, z = map(float, parts)
        return [x, y, z, 0]
    elif len(parts) == 4:
        x, y, z, t = map(float, parts)
        return [x, y, z, t]
    else:
        raise ValueError("Expected 3 or 4 values: x y z [t]")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive position control test for RoArm-M2."
    )
    parser.add_argument(
        "--port", type=str, default=None, help="Serial port (auto-detected if omitted)"
    )
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

    print("\nPosition Control Test")
    print("=====================")
    print("Enter coordinates as: x y z [t]")
    print("  x, y, z = position in mm")
    print("  t = end effector angle (optional, default 0)")
    print("Commands:")
    print("  home  - return to home position")
    print("  q     - quit")
    print()

    while True:
        try:
            line = input("Position> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line == "q" or line == "quit" or line == "exit":
            break

        if line == "home":
            print("Moving to home position...")
            arm.move_init()
            time.sleep(1)
            continue

        try:
            pos = parse_position(line)
            print(f"Moving to x={pos[0]}, y={pos[1]}, z={pos[2]}, t={pos[3]}...")
            arm.pose_ctrl(pos)
            time.sleep(0.5)
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Movement error: {e}")

    print("Returning to home position...")
    arm.move_init()
    time.sleep(2)
    print("Done.")


if __name__ == "__main__":
    main()
