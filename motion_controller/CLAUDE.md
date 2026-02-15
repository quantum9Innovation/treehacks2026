# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Context7
Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

# Self-Improvement

When you make a mistake or learn something important during a session, update CLAUDE.md with the lesson so you don't repeat it in future sessions.

# Bash Guidelines

## IMPORTANT: Avoid commands that cause output buffering issues
- DO NOT pipe output through `head`, `tail`, `less`, or `more` when monitoring or checking command output
- DO NOT use `| head -n X` or `| tail -n X` to truncate output - these cause buffering problems
- For log monitoring, prefer reading files directly rather than piping through filters

## When checking command output:
- Run commands directly without pipes when possible
- If you need to limit output, use command-specific flags (e.g., `git log -n 10` instead of `git log | head -10`)
- Avoid chained pipes that can cause output to buffer indefinitely

## Project

Robotics control project for the Waveshare RoArm M2 robotic arm. Scripts control the arm via USB serial using the `roarm-sdk` Python package.

## Commands

- **Run a script:** `uv run <script.py>`
- **Add a dependency:** `uv add <package>`
- **No test suite or linter configured**

## Hardware Setup

- **Arm model:** RoArm M2 (4-DOF)
- **Connection:** USB serial at `/dev/cu.usbserial-140`, 115200 baud
- **SDK import:** `from roarm_sdk.roarm import roarm`
- **Init:** `arm = roarm(roarm_type="roarm_m2", port="/dev/cu.usbserial-140", baudrate=115200)`

## RoArm M2 SDK Notes

### feedback_get() returns a list, not a dict
The SDK internally prints a dict with all fields (including torques) to stdout, but `feedback_get()` returns a list: `[x, y, z, base_rad, shoulder_rad, elbow_rad, hand_rad]`. Torque values are commented out in the SDK source (`common.py:handle_m2_feedback`).

### Getting torque values
Access torques via `arm.base_controller.base_data` after calling `feedback_get()`:
```python
arm.feedback_get()
d = arm.base_controller.base_data
# d has keys: torB, torS, torE, torH (values are 0.1% of stall torque, 1000 = 100%)
```

### Joint ranges (M2)
| Joint | ID | Radian range | Degree range |
|---|---|---|---|
| Base | 1 | [-π, π] | [-180, 180] |
| Shoulder | 2 | [-π/2, π/2] | [-90, 90] |
| Elbow | 3 | [-0.873, π] | [-50, 180] |
| Hand/Gripper | 4 | [0, π/2] | [0, 90] |

### Home position
Radians: `[0, 0, 1.5708, 0]` — use `arm.move_init()` (slow, speed=100) or `arm.joints_radian_ctrl(radians=[0, 0, 1.5708, 0], speed=4000, acc=200)` for fast homing.

### Joint coordinate system
- **Fully vertical** (arm pointing straight up): `[0, 0, 0, 0]` — both shoulder and elbow at 0
- **Fully horizontal** (arm pointing forward): shoulder=π/2, elbow=0
- **Home position** `[0, 0, π/2, 0]`: arm forward with elbow bent 90° down
- **Base joint** rotates around the z axis (yaw)
- **Shoulder** tilts the upper arm away from vertical (0=up, π/2=horizontal)
- **Elbow** angle is relative to the upper arm; negative values fold the forearm back toward vertical

### SDK elbow range wider than docs
The SDK actually validates elbow in [-1.2, 3.3] rad, which is wider than the [-0.873, π] range listed in hardware docs. Use -0.873 as the safe minimum.

### Torque-based collision detection
- A torque delta of **>50** on any joint between steps indicates likely contact with an object
- Use `abs(current_torque - previous_torque) > 50` as the collision threshold

### pose_ctrl limitations
- `pose_ctrl(pose=[x, y, z, gripper_angle])` — gripper_angle must be 0-90 degrees. The SDK validates this strictly.
- **Z must be 0-600mm** — pose_ctrl rejects negative Z values. Use `joints_radian_ctrl` for positions below the arm base.
- **pose_ctrl only moves the elbow** when lowering Z — it uses IK that bends the elbow while keeping the shoulder static. This does NOT produce meaningful torque changes for contact detection. For probing/contact detection, use `joints_radian_ctrl` to tilt the shoulder instead.

### Torque reading settling
After moving the arm to a new position, the first torque readings are unreliable (large deltas from gravity load redistribution, not contact). Always take a throwaway step and use its torques as the baseline before starting detection.

### Physical dimensions (RoArm M2)
- **Horizontal reach:** 316.15mm (from base center, fully extended)
- **Vertical height:** 236.8mm (from base to top of arm)
- **Base height:** 126.06mm (table to shoulder pivot)
- **End-effector length:** 13.29mm
- **Additional horizontal extension:** 30.00mm

### Arm geometry (IK/FK link lengths)
Derived from physical dimensions (reach + height specs):
- **L1 = 113mm** (shoulder to elbow)
- **L2 = 186mm** (elbow to end effector)
- **R_OFF = 17mm** (radial offset from base axis)
- **Z_OFF = -2mm** (vertical offset)
- **Max reach at ground level:** ~299mm (less than 316mm because arm must also extend downward)

**IMPORTANT:** The arm's `feedback_get()` XYZ uses its own internal FK which disagrees with our link lengths. Always use our own `fk()` on the feedback joint angles (fb[3:7]) for position calculations — never use fb[0:3] directly.

### motion.py
Ground-relative coordinate system. `Motion` class wraps the arm:
- `probe_ground(base_angle=π/2)` — mandatory before motion. Probes at given base angle (default: 90° left, away from work area). Tilts shoulder down with 90° elbow, detects contact via torE delta >50.
- `move_to(x, y, z)` — Z=0 is detected ground, positive is up. Uses our IK, verifies position with our FK.
- `probe_height_at(x, y, safe_z=100)` — smooth descent from safe_z toward ground, polling torE. Returns ground-relative Z of contact.
- `get_pose()` — returns ground-relative (x, y, z) using our FK on feedback joint angles.
- `home()` — returns to [0, 0, π/2, 0].

### heightmap.py
Scans a 9x9 grid (base angles -30° to +30°, radii 150-280mm) using `probe_height_at()`. Calibrates ground at 90° left first. Objects show up as positive heights in the output grid.

### Workspace limits
- **Max horizontal reach:** ~316mm (fully extended), ~299mm at ground level
- **Heightmap radius range:** 150-280mm (stays within ground-level reach)
- **Ground Z typically probes at ~-100mm** (native coords)
