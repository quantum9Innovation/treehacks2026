# Agent V2: Object-Aware Robot Arm Control

Agent V2 separates **perception** (what objects are in the scene) from **reasoning** (what to do with them) from **execution** (how to move there). The LLM never outputs raw coordinates — it references objects by ID and the motor engine handles coordinate conversion.

## Architecture

```
Camera (RGB + Depth)
       |
  [Vision Backend]  ← pluggable: YOLO, mock, etc.
       |
  List[DetectedObject]  (label, bbox, center pixel)
       |
  [Depth Enrichment]  ← RealSense depth at each object's center pixel
       |
  [Coordinate Transform]  ← camera 3D → arm 3D via calibration matrix
       |
  List[DetectedObject]  (now with arm_position_mm)
       |
  [LLM Reasoning]  ← receives annotated image + structured object list JSON
       |
  Action: goto(object_id=2)
       |
  [Motor Engine]  ← looks up object's arm coords, calls pose_ctrl()
```

## Quick Start

```bash
# Run with YOLO detection (default)
uv run vlm-agent-v2

# Run with mock detections (no camera/GPU needed)
uv run vlm-agent-v2 --vision mock

# Run calibration first (required for arm coordinates)
uv run vlm-agent-v2 --calibrate

# Specify YOLO model variant
uv run vlm-agent-v2 --yolo-model yolo11n-seg.pt
```

## Calibration

Before the arm can move to detected objects, you need to calibrate the camera-to-arm coordinate transform. This maps 3D points from the camera's frame of reference to the arm's frame of reference.

### Running Calibration

```bash
uv run vlm-agent-v2 --calibrate
```

This starts an interactive procedure:
1. The arm moves to 8 predefined positions spanning the workspace
2. At each position, an OpenCV window shows the live camera feed
3. Click on the arm tip/gripper in the image
4. Press Enter to confirm (or 's' to skip, 'r' to retry, 'q' to quit early)
5. After collecting points, an SVD solver computes the optimal rigid transform
6. Results are saved to `agent_v2/calibration_data.json`

**Quality guide:**
- RMSE < 10mm: Excellent
- RMSE < 20mm: Good (usable)
- RMSE > 20mm: Poor — recalibrate with more careful clicks

Calibration only needs to be done once per physical setup (camera + arm positions).

## Interactive Commands

Once running, the REPL supports:

| Command     | Description                              |
|-------------|------------------------------------------|
| `quit`      | Exit the agent                           |
| `home`      | Move arm to home position                |
| `pos`       | Show current arm position                |
| `calibrate` | Run calibration procedure                |
| `detect`    | Run perception only (no LLM), show results in OpenCV window |
| *(other)*   | Process as a natural language task       |

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--vision {yolo,mock}` | `yolo` | Vision backend |
| `--yolo-model` | `yolo11n.pt` | YOLO model file (auto-downloads) |
| `--calibration` | `agent_v2/calibration_data.json` | Path to calibration file |
| `--calibrate` | — | Run calibration and exit |
| `--model` | `gpt-4.1-mini` | OpenAI model ID |
| `--reasoning-effort` | `low` | Thinking effort (low/medium/high) |
| `--port` | auto-detect | Serial port for robot arm |
| `--z-offset` | `300` | Z offset for inverted mounting (mm) |
| `--no-invert-z` | — | Disable Z-axis inversion |
| `--auto-confirm` | — | Skip user confirmation (dangerous!) |
| `--debug` | — | Enable debug logging |

## Vision Backends

The vision system is pluggable via the `VisionBackend` abstract class.

### YOLO (`--vision yolo`)

Uses [ultralytics](https://github.com/ultralytics/ultralytics) YOLO models. COCO pretrained with 80 object classes (cups, bottles, phones, etc.). Weights auto-download on first use.

- Detection: `yolo11n.pt` (bounding boxes only)
- Segmentation: `yolo11n-seg.pt` (bounding boxes + masks)

### Mock (`--vision mock`)

Returns hardcoded detections — useful for testing the pipeline without a camera or GPU.

### Adding a New Backend

1. Create `agent_v2/vision/my_backend.py`
2. Subclass `VisionBackend` and implement `detect()` and `name()`
3. Add a CLI option in `run.py`

```python
from agent_v2.vision.base import VisionBackend
from agent_v2.models import DetectedObject, BBox

class MyBackend(VisionBackend):
    def detect(self, color_image, confidence_threshold=0.3, classes=None):
        # Your detection logic here
        # Return list of DetectedObject with object_id, label,
        # confidence, bbox, and center_px filled
        ...

    def name(self):
        return "my-backend"
```

## Coordinate Transform Pipeline

The transform pipeline converts pixel detections to arm-reachable 3D coordinates:

1. **Detection**: Vision backend produces bounding boxes with center pixels
2. **Depth lookup**: At each center pixel, read depth from the RealSense depth frame (median over 5x5 patch for noise robustness)
3. **Deprojection**: Using camera intrinsics, convert (pixel_x, pixel_y, depth) → 3D point in camera frame
4. **Rigid transform**: Apply calibration matrix (R, t) to convert camera-frame 3D → arm-frame 3D

The key is that depth frames are **aligned** to color frames (`rs.align(rs.stream.color)`), so depth pixels correspond exactly to color pixels.

## File Structure

```
agent_v2/
  __init__.py
  models.py                  # DetectedObject, SceneState, BBox, Point3D
  vision/
    __init__.py
    base.py                  # Abstract VisionBackend interface
    yolo_backend.py          # YOLO via ultralytics
    mock_backend.py          # Hardcoded detections for testing
  coordinate_transform.py    # Pixel→3D + camera→arm transform
  calibration.py             # Interactive calibration + SVD solver
  annotate.py                # Draw bboxes/labels on frames
  tools.py                   # LLM tool declarations
  prompts.py                 # System/task prompts
  agent.py                   # Main AgentV2 orchestrator
  run.py                     # CLI entry point
  calibration_data.json      # Saved calibration (generated)
```
