"""Gemini Robotics ER 1.5 vision wrapper for detection, pointing, and segmentation."""

import base64
import io
import json
import logging
import re
import time

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger("agent_v3.vision")

# Gemini ER uses a 0-1000 normalized coordinate system
GEMINI_COORD_SCALE = 1000

# Default camera resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def denormalize_point(
    y_norm: float, x_norm: float, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT
) -> tuple[int, int]:
    """Convert Gemini [y, x] normalized (0-1000) coords to pixel [x, y].

    Args:
        y_norm: Normalized Y coordinate (0-1000)
        x_norm: Normalized X coordinate (0-1000)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        (pixel_x, pixel_y) tuple
    """
    pixel_x = int(round(x_norm / GEMINI_COORD_SCALE * width))
    pixel_y = int(round(y_norm / GEMINI_COORD_SCALE * height))
    pixel_x = max(0, min(width - 1, pixel_x))
    pixel_y = max(0, min(height - 1, pixel_y))
    return pixel_x, pixel_y


def denormalize_box(
    box_2d: list[float], width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT
) -> tuple[int, int, int, int]:
    """Convert Gemini [ymin, xmin, ymax, xmax] normalized (0-1000) to pixel coords.

    Args:
        box_2d: [ymin, xmin, ymax, xmax] in 0-1000 scale
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        (x1, y1, x2, y2) pixel bounding box
    """
    ymin, xmin, ymax, xmax = box_2d
    x1 = int(round(xmin / GEMINI_COORD_SCALE * width))
    y1 = int(round(ymin / GEMINI_COORD_SCALE * height))
    x2 = int(round(xmax / GEMINI_COORD_SCALE * width))
    y2 = int(round(ymax / GEMINI_COORD_SCALE * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def decode_mask(
    mask_b64: str,
    bbox_px: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """Decode a base64 PNG mask and place it in the full image frame.

    The mask PNG covers only the bounding box region. We resize it to the bbox
    dimensions and place it on a full-frame boolean mask.

    Args:
        mask_b64: Base64-encoded PNG mask
        bbox_px: (x1, y1, x2, y2) pixel bounding box
        image_shape: (height, width, ...) of the full image

    Returns:
        Full-frame boolean mask (height, width)
    """
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

    x1, y1, x2, y2 = bbox_px
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)

    mask_resized = mask_img.resize((bbox_w, bbox_h), Image.NEAREST)
    mask_arr = np.array(mask_resized) > 127

    full_mask = np.zeros((image_shape[0], image_shape[1]), dtype=bool)
    full_mask[y1 : y1 + bbox_h, x1 : x1 + bbox_w] = mask_arr

    return full_mask


def parse_gemini_json(text: str) -> dict | list | None:
    """Parse JSON from Gemini response, stripping markdown code fences if present.

    Args:
        text: Raw response text from Gemini

    Returns:
        Parsed JSON object, or None if parsing fails
    """
    text = text.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Gemini JSON response: {text[:200]}")
        return None


class GeminiVision:
    """Wrapper around Gemini Robotics ER 1.5 for spatial vision tasks."""

    def __init__(self, client: genai.Client, model: str = "gemini-robotics-er-1.5-preview"):
        self.client = client
        self.model = model

    def _encode_image(self, image: np.ndarray) -> types.Part:
        """Encode a BGR numpy image as a PNG Part for Gemini."""
        _, png = cv2.imencode(".png", image)
        return types.Part.from_bytes(data=png.tobytes(), mime_type="image/png")

    def _call(self, image: np.ndarray, prompt: str) -> str:
        """Send an image + prompt to Gemini ER and return the text response."""
        image_part = self._encode_image(image)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text

    def detect_objects(self, image: np.ndarray, query: str = "all objects") -> list[dict]:
        """Detect objects in an image using natural language query.

        Args:
            image: BGR numpy array
            query: Natural language description of what to detect

        Returns:
            List of detections: [{"box_2d": [y1,x1,y2,x2], "label": "..."}]
            Coordinates are in pixel space (denormalized).
        """
        prompt = f'Detect {query}. Return a JSON list of objects with "box_2d" ([ymin,xmin,ymax,xmax] in 0-1000 scale) and "label" fields.'

        t0 = time.time()
        text = self._call(image, prompt)
        elapsed = time.time() - t0
        logger.info(f"detect_objects({query!r}) in {elapsed:.2f}s")

        results = parse_gemini_json(text)
        if not isinstance(results, list):
            logger.warning(f"detect_objects: expected list, got {type(results)}")
            return []

        # Denormalize bounding boxes
        h, w = image.shape[:2]
        for det in results:
            if "box_2d" in det:
                det["box_2d_px"] = denormalize_box(det["box_2d"], w, h)

        return results

    def point_at(self, image: np.ndarray, query: str) -> list[dict]:
        """Point at objects matching a natural language query.

        Args:
            image: BGR numpy array
            query: Natural language description of object to point at

        Returns:
            List of points: [{"point": [y, x], "label": "..."}]
            With added "pixel" key as (pixel_x, pixel_y).
        """
        prompt = f'Point at {query}. Return a JSON list with "point" ([y,x] in 0-1000 scale) and "label" fields.'

        t0 = time.time()
        text = self._call(image, prompt)
        elapsed = time.time() - t0
        logger.info(f"point_at({query!r}) in {elapsed:.2f}s")

        results = parse_gemini_json(text)
        if not isinstance(results, list):
            logger.warning(f"point_at: expected list, got {type(results)}")
            return []

        h, w = image.shape[:2]
        for pt in results:
            if "point" in pt:
                pt["pixel"] = denormalize_point(pt["point"][0], pt["point"][1], w, h)

        return results

    def segment(self, image: np.ndarray, query: str) -> list[dict]:
        """Segment objects matching a natural language query.

        Args:
            image: BGR numpy array
            query: Natural language description of object to segment

        Returns:
            List of segments: [{"box_2d": [...], "label": "...", "mask": "base64..."}]
            With added "box_2d_px", "mask_full" (boolean ndarray), and "centroid" keys.
        """
        prompt = f'Segment {query}. Return a JSON list with "box_2d" ([ymin,xmin,ymax,xmax] in 0-1000 scale), "label", and "mask" (base64-encoded PNG) fields.'

        t0 = time.time()
        text = self._call(image, prompt)
        elapsed = time.time() - t0
        logger.info(f"segment({query!r}) in {elapsed:.2f}s")

        results = parse_gemini_json(text)
        if not isinstance(results, list):
            logger.warning(f"segment: expected list, got {type(results)}")
            return []

        h, w = image.shape[:2]
        for seg in results:
            if "box_2d" in seg:
                bbox_px = denormalize_box(seg["box_2d"], w, h)
                seg["box_2d_px"] = bbox_px

                if "mask" in seg:
                    try:
                        mask_full = decode_mask(seg["mask"], bbox_px, image.shape)
                        seg["mask_full"] = mask_full

                        # Compute centroid from mask
                        ys, xs = np.where(mask_full)
                        if len(xs) > 0:
                            seg["centroid"] = (int(np.mean(xs)), int(np.mean(ys)))
                        else:
                            # Fallback to bbox center
                            seg["centroid"] = (
                                (bbox_px[0] + bbox_px[2]) // 2,
                                (bbox_px[1] + bbox_px[3]) // 2,
                            )
                    except Exception as e:
                        logger.warning(f"Failed to decode mask: {e}")
                        seg["centroid"] = (
                            (bbox_px[0] + bbox_px[2]) // 2,
                            (bbox_px[1] + bbox_px[3]) // 2,
                        )
                else:
                    # No mask returned, use bbox center as centroid
                    seg["centroid"] = (
                        (bbox_px[0] + bbox_px[2]) // 2,
                        (bbox_px[1] + bbox_px[3]) // 2,
                    )

        return results
