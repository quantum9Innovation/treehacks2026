# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context7
Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

## Self-Improvement
When you make a mistake or learn something important during a session, update CLAUDE.md with the lesson so you don't repeat it in future sessions.

## Project Overview

Vision-driven robot arm control system for the **RoArm M2** (4-DOF). Multiple LLM agent versions orchestrate a perception-action loop: camera captures scene, vision model detects/segments objects, LLM reasons about actions, motion controller executes movements.

## Commands

```bash
uv run vlm-agent                   # Run agent (multi-provider: OpenAI or Gemini)
uv run vlm-agent --calibrate       # Run camera-to-arm calibration
uv run vlm-agent --provider gemini # Use Gemini ER provider
uv add <package>                   # Add dependency
```

**Linting/formatting**:
```bash
uv run ruff check .          # Lint
uv run ruff format .         # Auto-format
uv run pyright               # Type check
```

Ruff ignores: F403, F405, E731. Pyright ignores missing imports.

## Architecture

### Agent (`agent/`)

Consolidated agent package. Multi-provider LLM orchestrator (OpenAI GPT or Gemini ER) with dual vision (SAM2 + Gemini ER). The agent calls tools (`detect`, `segment`, `goto_pixel`, `move_to_xyz`, `gripper_ctrl`, etc.) via structured function calling. Auto-injects camera frames each turn.

Key files:
- **`agent.py`**: CLI orchestrator (`AgentV3` class) — hardware init, agent loop, interactive REPL
- **`agent_wrapper.py`**: Web orchestrator (`WebAgentV3`) — wraps agent for web UI with EventBus
- **`llm.py`**: LLM provider abstraction (`OpenAIProvider`, `GeminiProvider`), `create_openai_client`, `ConfirmationHandler`
- **`tools.py`**: Tool declarations (provider-agnostic)
- **`prompts.py`**: System/task prompt generation
- **`camera.py`**: `RealSenseCamera` wrapper
- **`gemini_vision.py`**: Gemini ER vision (detect, point, segment)
- **`run.py`**: CLI entry point (`uv run vlm-agent`)

### Vision Pipeline

`camera → SAM2/Gemini ER detection → depth enrichment (RealSense) → coordinate transform → arm coordinates`

- **Vision backends** (`agent/vision/`): Abstract `VisionBackend` interface with SAM2 (default), YOLO, and mock implementations.
- **Coordinate transform** (`agent/coordinate_transform.py`): Pixel (x,y) + depth → camera 3D → arm 3D via calibration matrix.
- **Calibration** (`agent/calibration.py`): Interactive 8-point procedure using least-squares affine solver. Saved to `calibration_data.json`. RMSE < 20mm is good.
- **Data model** (`agent/models.py`): `DetectedObject`, `SceneState`, `Point3D` flow through the pipeline.

### Motion Controller (`motion_controller/`)

Ground-relative coordinate system (Z=0 at detected ground, positive up).

Key classes/methods in `motion_controller/motion.py`:
- `probe_ground(base_angle=π/2)` — mandatory before any motion. Probes at given base angle (default: 90° left, away from work area). Tilts shoulder down with 90° elbow, detects contact via torE delta >50.
- `move_to(x, y, z)` — Z=0 is detected ground, positive is up. Uses custom IK, verifies position with custom FK.
- `probe_height_at(x, y, safe_z=100)` — smooth descent from safe_z toward ground, polling torE. Returns ground-relative Z of contact.
- `get_pose()` — returns ground-relative (x, y, z) using custom FK on feedback joint angles.
- `home()` — returns to [0, 0, π/2, 0].

**`heightmap.py`**: Scans a 9x9 grid (base angles -30° to +30°, radii 150-280mm) using `probe_height_at()`. Calibrates ground at 90° left first. Objects show up as positive heights in the output grid.

### Hardware Setup

- **Arm model:** RoArm M2 (4-DOF), 4 arms on `/dev/ARM0`–`/dev/ARM3`
- **Connection:** USB serial, 115200 baud
- **SDK import:** `from roarm_sdk.roarm import roarm`
- **Camera**: Intel RealSense D435 at 640x480

### Arm Geometry (IK/FK)

- **L1 = 113mm** (shoulder to elbow)
- **L2 = 186mm** (elbow to end effector)
- **R_OFF = 17mm** (radial offset from base axis)
- **Z_OFF = -2mm** (vertical offset)
- **Horizontal reach:** 316mm (fully extended), ~299mm at ground level
- **Base height:** 126.06mm (table to shoulder pivot)
- **End-effector length:** 13.29mm
- **Heightmap radius range:** 150-280mm (stays within ground-level reach)

### RoArm M2 SDK Notes

#### feedback_get() returns a list, not a dict
The SDK prints a dict with all fields (including torques) to stdout, but `feedback_get()` returns a list: `[x, y, z, base_rad, shoulder_rad, elbow_rad, hand_rad]`. Torque values are commented out in the SDK source (`common.py:handle_m2_feedback`).

#### Getting torque values
Access torques via `arm.base_controller.base_data` after calling `feedback_get()`:
```python
arm.feedback_get()
d = arm.base_controller.base_data
# d has keys: torB, torS, torE, torH (values are 0.1% of stall torque, 1000 = 100%)
```

#### Joint ranges (M2)
| Joint | ID | Radian range | Degree range |
|---|---|---|---|
| Base | 1 | [-π, π] | [-180, 180] |
| Shoulder | 2 | [-π/2, π/2] | [-90, 90] |
| Elbow | 3 | [-0.873, π] | [-50, 180] |
| Hand/Gripper | 4 | [0, π/2] | [0, 90] |

SDK actually validates elbow in [-1.2, 3.3] rad (wider than docs). Use -0.873 as the safe minimum.

#### Home position
Radians: `[0, 0, 1.5708, 0]` — use `arm.move_init()` (slow, speed=100) or `arm.joints_radian_ctrl(radians=[0, 0, 1.5708, 0], speed=4000, acc=200)` for fast homing.

#### Joint coordinate system
- **Fully vertical** (arm pointing straight up): `[0, 0, 0, 0]` — both shoulder and elbow at 0
- **Fully horizontal** (arm pointing forward): shoulder=π/2, elbow=0
- **Home position** `[0, 0, π/2, 0]`: arm forward with elbow bent 90° down
- **Base joint** rotates around the z axis (yaw)
- **Shoulder** tilts the upper arm away from vertical (0=up, π/2=horizontal)
- **Elbow** angle is relative to the upper arm; negative values fold the forearm back toward vertical

#### pose_ctrl limitations
- `pose_ctrl(pose=[x, y, z, gripper_angle])` — gripper_angle must be 0-90 degrees (strictly validated).
- **Z must be 0-600mm** — rejects negative Z. Use `joints_radian_ctrl` for positions below the arm base.
- **Only moves the elbow** when lowering Z — uses IK that bends the elbow while keeping shoulder static. Does NOT produce meaningful torque changes for contact detection. Use `joints_radian_ctrl` to tilt shoulder instead.

### Critical Gotchas

- **Never use `feedback_get()[0:3]` for position** — the SDK's internal FK disagrees with actual link lengths. Always run custom `fk()` on joint angles (`feedback_get()[3:7]`).
- **First torque reading after movement is garbage** — gravity redistribution causes false deltas. Always discard it and use the second reading as baseline.
- **Torque collision threshold**: `abs(current_torque - previous_torque) > 50` on any joint indicates likely contact.
- **Z axis is inverted** in default arm config (designed for inverted mounting).

### Web Server (`server/`)

FastAPI backend serving REST + WebSocket APIs for the control panel.

- **Entry point**: `server/app.py` → `create_app()` with `lifespan` context manager
- **Config**: `server/config.py` — `Settings` class using `pydantic_settings`, reads `.env`
- **Hardware manager**: `server/hardware.py` — `HardwareManager` singleton owns camera, vision, arm, coordinate transform. Arm connection is deferred (not connected at boot).
- **Thread pools**: `_hw_executor` (single-thread, serializes arm/camera access), `_vision_executor` (single-thread, vision inference)
- **Arm lock**: `hw.arm_lock` (asyncio.Lock) serializes all arm movement commands
- **Routers**: `server/routers/` — `arm.py`, `camera.py`, `vision.py`, `agent.py`, `calibration.py`
- **Event bus**: `server/events.py` — `EventBus` publishes events over WebSocket

#### Arm startup behavior
- **Ground probe is NOT automatic** — server boots without probing ground. The arm just does `move_init()` on connect.
- Ground probe is triggered manually via `POST /api/arm/probe-ground` (button in control panel).
- Arm connection is also manual via the control panel (multi-device support: `/dev/ARM0`–`/dev/ARM3`).

#### Key arm endpoints (`/api/arm/`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/devices` | List available arms + active device |
| POST | `/connect` | Connect to arm `{"device": "/dev/ARM0"}` |
| POST | `/disconnect` | Disconnect current arm |
| GET | `/status` | Connection + ground calibration + active device |
| POST | `/probe-ground` | Probe ground to calibrate Z=0 |
| GET | `/pose` | Current ground-relative position |
| POST | `/move` | Move to (x, y, z) |
| POST | `/home` | Return to home position |
| POST | `/stop` | Emergency stop |
| POST | `/gripper` | Set gripper angle (0-90°) |
| POST | `/goto-pixel` | Move to 3D position from pixel coords |

### Web Frontend (`web/`)

React + TypeScript SPA (TanStack Router, Tailwind, shadcn/ui).

- **State**: `web/src/lib/robot-context.tsx` — `useReducer`-based global state (`useRobot()` hook)
- **API client**: `web/src/lib/api.ts` — typed wrappers around `fetch` (`armApi`, `visionApi`, `agentApi`, `calibrationApi`)
- **WebSocket**: `web/src/lib/hooks/use-websocket.ts` — auto-reconnecting WS that dispatches events to global state
- **Routes**: `web/src/routes/` — `index.tsx` (control page), `agent.tsx`, `calibration.tsx`
- **Control components**: `web/src/components/control/` — `movement-grid`, `gripper-control`, `position-readout`, `probe-ground`, `emergency-stop`, `step-size-selector`

## Environment Setup

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `HELICONE_API_KEY`. Python 3.12+ required. Uses `uv` as package manager with `hatchling` build backend.

## Bash Guidelines

- **Always use `uv run` to execute Python and project commands** — never use bare `python`, `ruff`, `pyright`, etc.
- Do not pipe output through `head`, `tail`, `less`, or `more` — causes buffering issues
- Use command-specific flags to limit output (e.g., `git log -n 10` not `git log | head -10`)
