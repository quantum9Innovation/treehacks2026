"""Draw bounding boxes and labels on frames for LLM consumption."""

import cv2
import numpy as np

from .models import DetectedObject

# Distinct color palette (BGR)
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (128, 255, 0),  # lime
    (0, 128, 255),  # orange
    (255, 0, 128),  # pink
    (128, 0, 255),  # purple
]


def annotate_frame(
    color_image: np.ndarray,
    detections: list[DetectedObject],
) -> np.ndarray:
    """Draw bounding boxes and labels on a frame.

    Args:
        color_image: BGR image (will be copied, not modified in place)
        detections: List of detected objects

    Returns:
        Annotated image with bounding boxes and labels
    """
    annotated = color_image.copy()

    for obj in detections:
        color = COLORS[obj.object_id % len(COLORS)]
        bbox = obj.bbox

        # Draw bounding box
        cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

        # Build label text
        label = f"[{obj.object_id}] {obj.label} ({obj.confidence:.2f})"

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(bbox.y1 - 6, text_h + 4)
        cv2.rectangle(
            annotated,
            (bbox.x1, label_y - text_h - 4),
            (bbox.x1 + text_w + 4, label_y + baseline),
            color,
            -1,
        )

        # Draw label text (black on colored background)
        cv2.putText(
            annotated,
            label,
            (bbox.x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        # Draw center point
        cv2.circle(annotated, obj.center_px, 4, color, -1)

    return annotated
