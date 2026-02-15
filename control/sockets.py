"""WebSocket client to read HMD pose data and control RoArm-M2 joint 4."""

import asyncio
import curses
import glob
import json
import math
import sys

import websockets
from roarm_sdk.roarm import roarm

# Zero position joint angles (in degrees)
ZERO_JOINT1 = 90
ZERO_JOINT2 = 0
ZERO_JOINT3 = 180
ZERO_JOINT4 = 0


def detect_serial_port():
    """Detect the serial port for the RoArm-M2."""
    SERIAL_GLOB_PATTERNS = [
        "/dev/cu.usbserial-*",
        "/dev/cu.usbmodem*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/ARM*",
    ]

    devices = []
    for pattern in SERIAL_GLOB_PATTERNS:
        devices.extend(glob.glob(pattern))

    return devices[0] if len(devices) == 1 else None


def normalize_angle_difference(current: float, zero: float) -> float:
    """
    Calculate the angular difference accounting for wraparound at ±180.

    Args:
        current: Current angle in degrees
        zero: Zero reference angle in degrees

    Returns:
        Angular difference in degrees, normalized to [-180, 180]
    """
    diff = current - zero
    # Normalize to [-180, 180]
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def clip_joint1_angle(yrot_degrees: float) -> float:
    """
    Clip yrot rotation to valid joint 1 range [-3.1415926, 3.1415926] radians.

    Joint 1 range: [-3.1415926, 3.1415926] radians (-180° to 180°)

    Args:
        yrot_degrees: Rotation in degrees from HMD

    Returns:
        Clipped angle in radians for joint 1
    """
    # Convert degrees to radians
    yrot_rad = math.radians(yrot_degrees)

    # Clip to [-3.1415926, 3.1415926]
    return max(-3.1415926, min(yrot_rad, 3.1415926))


def clip_joint3_angle(xrot_degrees: float) -> float:
    """
    Clip xrot rotation to valid joint 3 range [-0.8726646, 3.1415926] radians.

    Joint 3 range: [-0.8726646, 3.1415926] radians (approx -50° to 180°)

    Args:
        xrot_degrees: Rotation in degrees from HMD

    Returns:
        Clipped angle in radians for joint 3
    """
    # Convert degrees to radians
    xrot_rad = math.radians(xrot_degrees)

    # Clip to [-0.8726646, 3.1415926]
    return max(-0.8726646, min(xrot_rad, 3.1415926))


async def control_arm_from_pose(stdscr):
    """Connect to pose WebSocket and control arm joint 3 with xrot."""
    stdscr.nodelay(True)
    uri = "ws://10.19.182.3:8766"

    # Detect and connect to arm
    port = detect_serial_port()
    if not port:
        sys.exit(1)

    arm = roarm(roarm_type="roarm_m2", port=port, baudrate=115200)
    arm.echo_set(0)
    arm.torque_set(1)
    arm.move_init()

    # Zero reference values from HMD
    xrot_zero = 0.0
    yrot_zero = 0.0

    try:
        async with websockets.connect(uri) as websocket:
            # Auto-zero on startup
            # Get initial HMD position
            message = await websocket.recv()
            data = json.loads(message)

            # Store the zero reference angles
            xrot_zero = data["xrot"]
            yrot_zero = data["yrot"]

            # Zero the arm to preconfigured position
            arm.joint_radian_ctrl(1, math.radians(ZERO_JOINT1), 2048, 127)
            arm.joint_radian_ctrl(2, math.radians(ZERO_JOINT2), 2048, 127)
            arm.joint_radian_ctrl(3, math.radians(ZERO_JOINT3), 2048, 127)
            arm.joint_radian_ctrl(4, math.radians(ZERO_JOINT4), 2048, 127)

            while True:
                message = await websocket.recv()
                data = json.loads(message)

                # Check for spacebar press
                try:
                    key = stdscr.getkey()
                    if key == " ":
                        # Store new zero reference angles
                        xrot_zero = data["xrot"]
                        yrot_zero = data["yrot"]

                        # Zero the arm to preconfigured position
                        arm.joint_radian_ctrl(1, math.radians(ZERO_JOINT1), 2048, 127)
                        arm.joint_radian_ctrl(2, math.radians(ZERO_JOINT2), 2048, 127)
                        arm.joint_radian_ctrl(3, math.radians(ZERO_JOINT3), 2048, 127)
                        arm.joint_radian_ctrl(4, math.radians(ZERO_JOINT4), 2048, 127)
                except curses.error:
                    pass

                # Calculate angle differences accounting for wraparound
                xrot_diff = normalize_angle_difference(data["xrot"], xrot_zero)
                yrot_diff = normalize_angle_difference(data["yrot"], yrot_zero)

                # Apply transformations and add zero position offsets
                xrot_degrees = -xrot_diff - 90 + ZERO_JOINT3
                joint3_angle = clip_joint3_angle(xrot_degrees)
                arm.joint_radian_ctrl(3, joint3_angle, 2048, 127)

                yrot_degrees = -yrot_diff - 90 + ZERO_JOINT1
                joint1_angle = clip_joint1_angle(yrot_degrees)
                arm.joint_radian_ctrl(1, joint1_angle, 2048, 127)

    except (
        websockets.exceptions.ConnectionClosed,
        ConnectionRefusedError,
        KeyboardInterrupt,
        Exception,
    ):
        arm.move_init()


async def read_pose_stream():
    """Connect to the pose WebSocket and print data in real time."""
    uri = "ws://10.19.182.3:8766"

    try:
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                print(
                    f"Position: x={data['x']:7.3f}, y={data['y']:7.3f}, z={data['z']:7.3f} | "
                    f"Rotation: xrot={data['xrot']:7.2f}, yrot={data['yrot']:7.2f}, zrot={data['zrot']:7.2f} | "
                    f"Valid: {data['valid']}"
                )

    except (
        websockets.exceptions.ConnectionClosed,
        ConnectionRefusedError,
        KeyboardInterrupt,
        Exception,
    ):
        pass


def main(stdscr):
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--control":
        # Control arm mode
        asyncio.run(control_arm_from_pose(stdscr))
    else:
        # Print only mode
        asyncio.run(read_pose_stream())


if __name__ == "__main__":
    curses.wrapper(main)
