"""Vision endpoints: look, segment, and detect."""

import base64
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..hardware import HardwareManager

logger = logging.getLogger("server.routers.vision")
router = APIRouter(prefix="/api/vision", tags=["vision"])


class SegmentRequest(BaseModel):
    pixel_x: int = Field(ge=0, le=639)
    pixel_y: int = Field(ge=0, le=479)


class DeprojectRequest(BaseModel):
    pixel_x: int = Field(ge=0, le=639)
    pixel_y: int = Field(ge=0, le=479)


class DetectRequest(BaseModel):
    query: str


def _encode_b64(image: np.ndarray, quality: int = 85) -> str:
    _, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(jpg.tobytes()).decode()


@router.post("/look")
async def look(request: Request):
    """Capture a frame and optionally encode for SAM2. Returns base64 images."""
    hw: HardwareManager = request.app.state.hardware
    if hw.camera is None or hw.ct is None:
        raise HTTPException(503, "Hardware not ready")

    def _capture_and_encode():
        frames = hw.camera.pipeline.wait_for_frames()
        aligned = hw.ct.get_aligned_frames(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")

        color_image = np.asanyarray(color_frame.get_data())
        colorized = hw.camera.colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized.get_data())
        depth_image = hw.camera._draw_depth_scale(
            depth_image.copy(), hw.camera.min_depth_m, hw.camera.max_depth_m
        )

        # Cache for subsequent segment/detect/goto calls
        hw.vision_color = color_image
        hw.vision_depth_frame = depth_frame

        # Encode for SAM2 if available
        encode_time = 0.0
        if hw.sam2 is not None:
            t0 = time.time()
            hw.sam2.set_image(color_image)
            encode_time = time.time() - t0

        return color_image, depth_image, encode_time

    color_image, depth_image, encode_time = await hw.run_in_hw_thread(
        _capture_and_encode
    )

    result = {
        "color_b64": _encode_b64(color_image),
        "depth_b64": _encode_b64(depth_image),
        "width": 640,
        "height": 480,
    }
    if hw.sam2 is not None:
        result["sam2_encode_time_s"] = round(encode_time, 2)

    return result


@router.post("/segment")
async def segment(body: SegmentRequest, request: Request):
    """Run SAM2 point-prompt segmentation at a pixel coordinate."""
    hw: HardwareManager = request.app.state.hardware
    if hw.sam2 is None:
        raise HTTPException(503, "SAM2 vision not available")
    if hw.ct is None:
        raise HTTPException(503, "Coordinate transform not ready")
    if hw.vision_color is None:
        raise HTTPException(400, "Call /api/vision/look first to capture a frame")

    # Run segmentation
    def _segment():
        mask, score, bbox = hw.sam2.segment_point(body.pixel_x, body.pixel_y)
        return mask, float(score), [int(b) for b in bbox]

    mask, score, bbox = await hw.run_in_vision_thread(_segment)

    # Compute arm coordinates from pixel
    arm_coords = None
    depth_mm = None
    if hw.vision_depth_frame is not None:
        depth_mm_raw = hw.ct.get_depth_at_pixel(
            hw.vision_depth_frame, body.pixel_x, body.pixel_y
        )
        if depth_mm_raw is not None and depth_mm_raw > 0:
            depth_mm = round(float(depth_mm_raw), 1)
            cam_3d = hw.ct.deproject_pixel(
                body.pixel_x, body.pixel_y, depth_mm=depth_mm_raw
            )
            if cam_3d is not None:
                arm_3d = hw.ct.camera_to_arm(cam_3d)
                if arm_3d is not None:
                    arm_coords = {
                        "x": round(float(arm_3d[0]), 1),
                        "y": round(float(arm_3d[1]), 1),
                        "z": round(float(arm_3d[2]), 1),
                    }

    # Generate annotated overlay
    annotated = hw.vision_color.copy()
    # Green mask overlay
    mask_overlay = np.zeros_like(annotated)
    mask_overlay[mask] = (0, 200, 0)
    annotated = cv2.addWeighted(annotated, 1.0, mask_overlay, 0.4, 0)
    # Bbox
    cv2.rectangle(
        annotated,
        (bbox[0], bbox[1]),
        (bbox[2], bbox[3]),
        (0, 255, 0),
        2,
    )
    # Click point
    cv2.drawMarker(
        annotated,
        (body.pixel_x, body.pixel_y),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        15,
        2,
    )

    return {
        "score": round(score, 3),
        "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
        "mask_area_px": int(mask.sum()),
        "depth_mm": depth_mm,
        "arm_coordinates": arm_coords,
        "annotated_b64": _encode_b64(annotated),
    }


@router.post("/detect")
async def detect(body: DetectRequest, request: Request):
    """Run Gemini ER detection with a natural language query."""
    hw: HardwareManager = request.app.state.hardware
    if hw.gemini_vision is None:
        raise HTTPException(503, "Gemini Vision not available")
    if hw.ct is None:
        raise HTTPException(503, "Coordinate transform not ready")
    if hw.vision_color is None:
        raise HTTPException(400, "Call /api/vision/look first to capture a frame")

    def _detect():
        return hw.gemini_vision.segment(hw.vision_color, body.query)

    segments = await hw.run_in_vision_thread(_detect)

    if not segments:
        return {
            "status": "error",
            "message": f"No objects matching '{body.query}' detected",
        }

    seg = segments[0]
    bbox_px = seg.get("box_2d_px", (0, 0, 0, 0))
    label = seg.get("label", body.query)
    centroid = seg.get(
        "centroid", ((bbox_px[0] + bbox_px[2]) // 2, (bbox_px[1] + bbox_px[3]) // 2)
    )
    cx, cy = centroid

    # Compute arm coordinates
    arm_coords = None
    depth_mm = None
    if hw.vision_depth_frame is not None:
        depth_mm_raw = hw.ct.get_depth_at_pixel(hw.vision_depth_frame, cx, cy)
        if depth_mm_raw is not None and depth_mm_raw > 0:
            depth_mm = round(float(depth_mm_raw), 1)
            cam_3d = hw.ct.deproject_pixel(cx, cy, depth_mm=depth_mm_raw)
            if cam_3d is not None:
                arm_3d = hw.ct.camera_to_arm(cam_3d)
                if arm_3d is not None:
                    arm_coords = {
                        "x": round(float(arm_3d[0]), 1),
                        "y": round(float(arm_3d[1]), 1),
                        "z": round(float(arm_3d[2]), 1),
                    }

    mask = seg.get("mask_full")
    mask_area = int(mask.sum()) if mask is not None else 0

    # Annotated overlay
    annotated = hw.vision_color.copy()
    if mask is not None and mask.any():
        overlay = annotated.copy()
        overlay[mask] = [0, 200, 0]
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
    cv2.rectangle(
        annotated, (bbox_px[0], bbox_px[1]), (bbox_px[2], bbox_px[3]), (0, 255, 0), 2
    )
    cv2.circle(annotated, (cx, cy), 6, (255, 0, 0), -1)

    return {
        "query": body.query,
        "label": label,
        "centroid": {"pixel_x": cx, "pixel_y": cy},
        "bbox": {
            "x1": bbox_px[0],
            "y1": bbox_px[1],
            "x2": bbox_px[2],
            "y2": bbox_px[3],
        },
        "mask_area_px": mask_area,
        "depth_mm": depth_mm,
        "arm_coordinates": arm_coords,
        "annotated_b64": _encode_b64(annotated),
    }


@router.post("/deproject")
async def deproject(body: DeprojectRequest, request: Request):
    """Get 3D coordinates from a pixel."""
    hw: HardwareManager = request.app.state.hardware
    if hw.ct is None:
        raise HTTPException(503, "Coordinate transform not ready")

    depth_frame = hw.vision_depth_frame
    if depth_frame is None:
        bundle = hw.latest_frame
        if bundle is None:
            raise HTTPException(400, "No frame available")
        depth_frame = bundle.depth_frame

    depth_mm = hw.ct.get_depth_at_pixel(depth_frame, body.pixel_x, body.pixel_y)
    if depth_mm is None or depth_mm <= 0:
        raise HTTPException(400, f"No valid depth at ({body.pixel_x}, {body.pixel_y})")

    cam_3d = hw.ct.deproject_pixel(body.pixel_x, body.pixel_y, depth_mm=depth_mm)
    if cam_3d is None:
        raise HTTPException(400, "Deprojection failed")

    arm_3d = None
    if hw.ct._M is not None:
        result = hw.ct.camera_to_arm(cam_3d)
        if result is not None:
            arm_3d = {
                "x": round(float(result[0]), 1),
                "y": round(float(result[1]), 1),
                "z": round(float(result[2]), 1),
            }

    return {
        "depth_mm": round(float(depth_mm), 1),
        "camera_3d": {
            "x": round(float(cam_3d[0]), 1),
            "y": round(float(cam_3d[1]), 1),
            "z": round(float(cam_3d[2]), 1),
        },
        "arm_3d": arm_3d,
    }
