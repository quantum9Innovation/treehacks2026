# Multi-Agent System for 3-Arm Collaborative Manipulation

## Context

The robot has 4 physical arms (`/dev/ARM0`-`ARM3`), a shared Intel RealSense D435 camera, and an existing single-arm agent (`agent_v3/`). The server already supports multiple simultaneous arm connections with per-device Motion instances and asyncio locks (`server/hardware.py`), but the **agent is a singleton** and the **single-threaded `_hw_executor` prevents true parallel arm motion**. We need a hierarchical multi-agent system where a coordinator LLM plans collaborative tasks and dispatches subtasks to 3 per-arm worker agents that execute in parallel.

## Architecture

```
                    ┌─────────────────────────┐
                    │    Coordinator Agent     │
                    │  (scene understanding,   │
                    │   task planning, sync)   │
                    └───┬───────┬───────┬──────┘
                        │       │       │
                  ┌─────▼──┐ ┌──▼────┐ ┌▼──────┐
                  │Worker 0│ │Worker 1│ │Worker 2│
                  │ (ARM0) │ │ (ARM1) │ │ (ARM2) │
                  └───┬────┘ └───┬────┘ └───┬────┘
                      │          │          │
                  ┌───▼────┐ ┌───▼────┐ ┌───▼────┐
                  │Motion 0│ │Motion 1│ │Motion 2│
                  └────────┘ └────────┘ └────────┘
```

- **Coordinator**: LLM agent (GPT/Gemini via existing `LLMProvider`) that sees the camera feed, detects objects, and assigns subtasks to arms. It orchestrates via phases — parallel subtasks within a phase, barriers between phases.
- **Workers**: Lightweight LLM agents (one per arm) that execute a single subtask using motion/vision tools. Reuse `WebAgentV3`'s tool executor methods but bound to a specific device.

## Implementation Plan

### Step 1: Enable parallel arm motion in `server/hardware.py`

Currently all arm commands serialize through one thread (`_hw_executor`). Add per-device executors so arms can move simultaneously.

**Changes**:
- Add `_arm_executors: dict[str, ThreadPoolExecutor]` to `HardwareManager`
- Create a per-device executor in `_connect_arm_sync()`, shut it down in `_disconnect_arm_sync()`
- Add `run_in_arm_thread(device, fn, *args)` method
- Keep `run_in_hw_thread` for non-arm operations (camera init, etc.)

### Step 2: Per-device coordinate transforms in `server/hardware.py`

Currently `self.ct` is shared and `load_calibration_for_device()` overwrites the single matrix. For parallel operation, each arm needs its own `CoordinateTransform`.

**Changes**:
- Add `_per_device_ct: dict[str, CoordinateTransform]` to `HardwareManager`
- In `_connect_arm_sync()`, create a new `CoordinateTransform` per device with device-specific calibration
- Add `get_ct(device)` accessor method
- Keep `self.ct` as the active-device shortcut for backward compat

### Step 3: Create `agent_multiarm/` package with data models

New package: `agent_multiarm/`

**`agent_multiarm/models.py`** — Core data structures:
- `CoordinatorState` enum: `idle | planning | executing_phase | replanning | done | error`
- `WorkerState` enum: `idle | executing | done | error`
- `SubtaskResult` dataclass: `device, status, message, final_position, error`
- `PhaseResult` dataclass: `phase_index, all_succeeded, subtask_results`
- `MultiArmSession` dataclass: overall session tracking

### Step 4: Extract shared tool executors into `agent_multiarm/tool_executor.py`

`WebAgentV3` and the new workers both need the same tool executor methods (`_execute_goto_pixel`, `_execute_move_to_xyz`, etc.). Extract them into a mixin/base class parameterized by `(motion, ct, hw)` so both can reuse without duplication.

**Key change**: All executors use `self._motion` and `self._ct` instead of `self._hw.motion` / `self._hw.ct`, allowing per-device binding.

**Files to refactor**:
- `server/agent_wrapper.py` — have `WebAgentV3` inherit/compose the shared executor
- New `agent_multiarm/tool_executor.py` — the extracted base class

### Step 5: Implement worker agent (`agent_multiarm/worker.py`)

`WorkerAgent` — a single-arm LLM agent that executes one subtask.

**Design**:
- Constructor: `(device, hw, provider, bus)` — bound to a specific arm
- Uses tool executor base class with device-specific `motion` and `ct`
- Simplified system prompt (focused on single subtask execution, not open-ended reasoning)
- Max 6 iterations (subtasks are simpler than full tasks)
- No confirmation flow (coordinator already approved the plan)
- Returns `SubtaskResult` on completion
- Tool subset: all motion tools + `segment` (no `detect` — coordinator handles scene understanding)

### Step 6: Define coordinator tools (`agent_multiarm/coordinator_tools.py`)

The coordinator has **orchestration tools**, not motion tools:

| Tool | Purpose |
|------|---------|
| `observe_scene` | Capture camera frame + get all arm positions |
| `detect_object` | Natural language detection (delegates to Gemini ER) |
| `get_all_arm_poses` | Current position of all 3 arms |
| `assign_subtask` | Send subtask to a specific arm's worker (non-blocking) |
| `wait_for_phase` | Block until all assigned subtasks complete (barrier) |
| `emergency_stop_all` | Freeze all arms immediately |
| `home_all` | Send all arms to home |

### Step 7: Implement coordinator agent (`agent_multiarm/coordinator.py`)

`CoordinatorAgent` — the orchestrating LLM agent.

**Design**:
- Owns 3 `WorkerAgent` instances (one per connected arm)
- Coordinator LLM loop: observe scene -> plan phases -> assign subtasks -> wait -> repeat
- `assign_subtask` launches worker in per-arm executor thread, stores future
- `wait_for_phase` gathers all pending futures (barrier between phases)
- On failure: re-observe scene, re-plan remaining work
- Collision avoidance at planning level: coordinator prompt instructs LLM to never assign two arms to the same spatial region in the same phase
- Optional `constraints` on `assign_subtask` (forbidden base angle ranges, max Z)

**Coordinator system prompt** includes:
- Workspace layout (3 arms, positions, reach limits)
- Collision safety rules (spatial separation, height staggering, phase discipline)
- Planning pattern (observe -> detect -> assign -> wait -> repeat)

### Step 8: Server integration (`server/routers/agent_multiarm.py`)

New FastAPI router: `/api/multiarm/`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/task` | Submit a collaborative multi-arm task |
| GET | `/status` | Coordinator state + all worker states |
| POST | `/stop` | Emergency stop all arms + cancel task |
| POST | `/confirm` | Approve/reject if confirmation needed |

Register in `server/app.py` alongside existing routers.

**Event bus events** for WebSocket:
- `multiarm.coordinator.state` — planning/executing/done
- `multiarm.worker.state` — per-arm status + current subtask
- `multiarm.phase.complete` — phase results
- `multiarm.task.complete` — final result

### Step 9: Web UI — multi-arm orchestration page

New route: `web/src/routes/multiarm.tsx`

**Layout**: Camera feed (left) | Coordinator chat (center) | Per-arm status cards (right, 3 cards showing device, state, subtask, position)

**Changes**:
- Add state to `robot-context.tsx`: `coordinatorState`, `workerStates`, `currentPhase`
- Add WebSocket event handlers for `multiarm.*` events
- Add navigation link to new route

### Step 10: Entry point and wiring

- Add `agent_multiarm/__init__.py` with `CoordinatorAgent` export
- Add pyproject.toml entry point: `vlm-multiarm = "agent_multiarm.run:main"` for CLI usage
- Wire up in `server/app.py` lifespan if needed

## Key Files

| File | Action |
|------|--------|
| `server/hardware.py` | Modify — per-device executors + per-device CT |
| `server/agent_wrapper.py` | Refactor — extract tool executors into shared base |
| `agent_multiarm/__init__.py` | New |
| `agent_multiarm/models.py` | New — data structures |
| `agent_multiarm/tool_executor.py` | New — shared tool executor base |
| `agent_multiarm/worker.py` | New — per-arm worker agent |
| `agent_multiarm/coordinator_tools.py` | New — coordinator tool declarations |
| `agent_multiarm/coordinator_prompts.py` | New — coordinator system prompt |
| `agent_multiarm/coordinator.py` | New — coordinator agent |
| `agent_multiarm/run.py` | New — CLI entry point |
| `server/routers/agent_multiarm.py` | New — REST endpoints |
| `server/app.py` | Modify — register new router |
| `web/src/routes/multiarm.tsx` | New — UI page |
| `web/src/lib/robot-context.tsx` | Modify — add multiarm state |

## Reuse from existing code

- `agent_v3/llm.py` — `LLMProvider` (OpenAI/Gemini) used by both coordinator and workers
- `agent_v3/tools.py` — `get_tool_declarations()` for worker tool subset
- `server/agent_wrapper.py` — All tool executor methods extracted into shared base
- `server/events.py` — `EventBus` for real-time status publishing
- `server/routers/arm.py:_estop_one()` — Emergency stop logic for all arms
- `agent_v2/calibration.py:calibration_path_for_device()` — Per-arm calibration paths

## Verification

1. **Unit test parallel motion**: Connect 2+ arms, send simultaneous `/api/arm/move?device=` requests, verify they execute in parallel (not serialized)
2. **Worker test**: Submit a simple subtask to one `WorkerAgent`, verify it moves the correct arm and returns `SubtaskResult`
3. **Coordinator test**: Submit "ARM0 picks up X, ARM1 picks up Y" — verify coordinator creates 2 parallel subtasks, both execute, coordinator reports success
4. **Collision test**: Submit a task where arms would conflict — verify coordinator stages them into sequential phases
5. **E-stop test**: During multi-arm execution, hit stop — verify all arms freeze
6. **Web UI test**: Submit task via `/multiarm` page, verify live status updates for coordinator + all 3 workers via WebSocket
