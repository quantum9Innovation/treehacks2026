"""VR headset rotation → arm joint control. Yaw rotates base, pitch tilts arm."""

import argparse
import asyncio
import json
import logging
import math
import os
import select
import sys
import termios
import threading
import tty

import websockets

# Suppress noisy SDK serial logging
logging.getLogger().setLevel(logging.CRITICAL)

from motion_controller.motion import Motion

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
WS_URI = "ws://10.19.176.234:8766"

# Home joint positions (radians) — arm starts here, HMD zero maps to this
HOME_BASE = 0.0
HOME_SHOULDER = math.radians(-75)
HOME_ELBOW = math.pi / 2 + math.radians(75)  # compensate for shoulder tilt

# Scaling: HMD degrees → arm degrees (< 1.0 = reduced sensitivity)
YAW_SCALE = 1.0
PITCH_SCALE = 0.6

# Safe angle limits for the output joints (radians)
BASE_MIN = math.radians(-120)
BASE_MAX = math.radians(120)
ELBOW_MIN = HOME_ELBOW - math.radians(77)
ELBOW_MAX = HOME_ELBOW + math.radians(20)

MOVE_SPEED = 4000
MOVE_ACC = 200

# Minimum angle change (radians) to send a command — reduces jitter
MIN_ANGLE_RAD = math.radians(0.5)

# ---------------------------------------------------------------------------
# Keyboard listener (space to re-zero, works over SSH)
# ---------------------------------------------------------------------------
space_pressed = threading.Event()


def keyboard_listener():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch == " ":
                    space_pressed.set()
                elif ch == "\x03":  # Ctrl+C
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def normalize_angle_difference(current: float, zero: float) -> float:
    """Angular difference in degrees, normalized to [-180, 180]."""
    diff = current - zero
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


# ---------------------------------------------------------------------------
# Joint command
# ---------------------------------------------------------------------------
_devnull = open(os.devnull, "w")


def send_joints(motion: Motion, base, elbow):
    """Send joint angles directly, no settle wait. Shoulder is fixed."""
    base = max(BASE_MIN, min(BASE_MAX, base))
    elbow = max(ELBOW_MIN, min(ELBOW_MAX, elbow))

    old_stdout = sys.stdout
    sys.stdout = _devnull
    try:
        motion._ctrl_joints(radians=[base, HOME_SHOULDER, elbow, 0], speed=MOVE_SPEED, acc=MOVE_ACC)
    finally:
        sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

async def control_loop(motion: Motion, ws_uri: str):
    """Connect to VR pose WebSocket and map head rotation to arm joints."""
    threading.Thread(target=keyboard_listener, daemon=True).start()

    print(f"Connecting to VR pose stream: {ws_uri}")
    print("Press SPACE to re-zero, Ctrl+C to exit")

    last_base, last_elbow = None, None

    async with websockets.connect(ws_uri) as ws:
        # First message → zero reference
        data = json.loads(await ws.recv())
        xrot_zero = data["xrot"]
        yrot_zero = data["yrot"]
        print(f"Zeroed at xrot={xrot_zero:.1f} yrot={yrot_zero:.1f}")

        # Move arm to home position
        send_joints(motion, HOME_BASE, HOME_ELBOW)

        msg_count = 0
        cmd_count = 0
        skip_count = 0

        while True:
            data = json.loads(await ws.recv())
            msg_count += 1

            if not data.get("valid", False):
                if msg_count % 30 == 0:
                    print(f"[{msg_count}] invalid pose, skipping")
                continue

            # Re-zero on space
            if space_pressed.is_set():
                space_pressed.clear()
                xrot_zero = data["xrot"]
                yrot_zero = data["yrot"]
                send_joints(motion, HOME_BASE, HOME_ELBOW)
                last_base, last_elbow = HOME_BASE, HOME_ELBOW
                print(f"Re-zeroed at xrot={xrot_zero:.1f} yrot={yrot_zero:.1f}")
                continue

            # Compute deltas in degrees
            pitch_delta = normalize_angle_difference(data["xrot"], xrot_zero)
            yaw_delta = normalize_angle_difference(data["yrot"], yrot_zero)

            # Map to joint angles — deltas are relative to home position
            base = HOME_BASE + math.radians(-yaw_delta * YAW_SCALE)
            elbow = HOME_ELBOW + math.radians(pitch_delta * PITCH_SCALE)

            # Skip tiny changes
            if last_base is not None:
                if abs(base - last_base) < MIN_ANGLE_RAD and abs(elbow - last_elbow) < MIN_ANGLE_RAD:
                    skip_count += 1
                    continue

            send_joints(motion, base, elbow)
            cmd_count += 1
            last_base, last_elbow = base, elbow

            line = (
                f"[msg:{msg_count} cmd:{cmd_count} skip:{skip_count}] "
                f"yaw={yaw_delta:+.1f} pitch={pitch_delta:+.1f} -> "
                f"base={math.degrees(base):+.1f} elbow={math.degrees(elbow):+.1f}"
            )
            print(f"\033[2K\r{line[:200]}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="VR headset → arm joint control")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--ws-uri", default=WS_URI, help=f"VR pose WebSocket URI (default: {WS_URI})")
    parser.add_argument("--base-offset", type=float, default=0.0, help="Base joint offset in degrees")
    args = parser.parse_args()

    # Per-arm base rotation offset (match server/hardware.py)
    _base_offsets = {"ARM2": math.pi}
    if args.port:
        import pathlib
        arm_name = pathlib.Path(args.port).name
        base_off = _base_offsets.get(arm_name, 0.0)
    else:
        base_off = 0.0
    base_off += math.radians(args.base_offset)

    # Arms are mounted upside-down (inverted=True)
    motion = Motion(port=args.port, inverted=True, base_offset=base_off)

    print("Probing ground...")
    motion.probe_ground()
    print("Ground calibrated. Starting VR control.")

    try:
        asyncio.run(control_loop(motion, args.ws_uri))
    except KeyboardInterrupt:
        pass
    finally:
        print("Homing arm...")
        motion.home()


if __name__ == "__main__":
    main()
