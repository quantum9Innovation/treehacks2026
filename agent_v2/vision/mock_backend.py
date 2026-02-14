"""Mock vision backend returning hardcoded detections for testing."""

import numpy as np

from ..models import BBox, DetectedObject
from .base import VisionBackend


class MockBackend(VisionBackend):
    """Returns hardcoded detections for testing without camera/GPU."""

    def detect(
        self,
        color_image: np.ndarray,
        confidence_threshold: float = 0.3,
        classes: list[str] | None = None,
    ) -> list[DetectedObject]:
        h, w = color_image.shape[:2]

        detections = [
            DetectedObject(
                object_id=0,
                label="cup",
                confidence=0.95,
                bbox=BBox(x1=w // 4, y1=h // 3, x2=w // 4 + 80, y2=h // 3 + 100),
                center_px=(w // 4 + 40, h // 3 + 50),
            ),
            DetectedObject(
                object_id=1,
                label="bottle",
                confidence=0.88,
                bbox=BBox(
                    x1=w // 2 + 50, y1=h // 4, x2=w // 2 + 120, y2=h // 4 + 150
                ),
                center_px=(w // 2 + 85, h // 4 + 75),
            ),
            DetectedObject(
                object_id=2,
                label="cell phone",
                confidence=0.82,
                bbox=BBox(
                    x1=w // 2 - 40, y1=h // 2 + 30, x2=w // 2 + 30, y2=h // 2 + 90
                ),
                center_px=(w // 2 - 5, h // 2 + 60),
            ),
        ]

        if classes is not None:
            detections = [d for d in detections if d.label in classes]

        detections = [d for d in detections if d.confidence >= confidence_threshold]

        # Re-assign object_ids after filtering
        for i, d in enumerate(detections):
            d.object_id = i

        return detections

    def name(self) -> str:
        return "mock"
