"""YOLO vision backend using ultralytics."""

import logging

import numpy as np
from ultralytics import YOLO

from ..models import BBox, DetectedObject
from .base import VisionBackend

logger = logging.getLogger("agent_v2.vision.yolo")


class YOLOBackend(VisionBackend):
    """Object detection via YOLO (ultralytics).

    Supports both detection (yolo11n.pt) and segmentation (yolo11n-seg.pt) models.
    COCO pretrained with 80 classes including cups, bottles, phones, etc.
    Weights auto-download on first use.
    """

    def __init__(self, model_path: str = "yolo11n.pt"):
        logger.info(f"Loading YOLO model: {model_path}")
        self._model = YOLO(model_path)
        self._model_path = model_path
        self._is_seg = "-seg" in model_path
        logger.info(f"YOLO model loaded (segmentation={self._is_seg})")

    def detect(
        self,
        color_image: np.ndarray,
        confidence_threshold: float = 0.3,
        classes: list[str] | None = None,
    ) -> list[DetectedObject]:
        # Run inference
        results = self._model(color_image, conf=confidence_threshold, verbose=False)

        if not results or len(results) == 0:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Get class names from model
        names = result.names  # {0: 'person', 1: 'bicycle', ...}

        # Get masks if segmentation model
        masks = result.masks if self._is_seg else None

        detections = []
        idx = 0

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            label = names[cls_id]
            confidence = float(boxes.conf[i].item())

            # Filter by class names if specified
            if classes is not None and label not in classes:
                continue

            # Bounding box
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox = BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))

            # Mask (if available)
            mask = None
            if masks is not None:
                mask = masks.data[i].cpu().numpy()

            obj = DetectedObject(
                object_id=idx,
                label=label,
                confidence=confidence,
                bbox=bbox,
                center_px=bbox.center,
                mask=mask,
            )
            detections.append(obj)
            idx += 1

        logger.debug(f"Detected {len(detections)} objects")
        return detections

    def name(self) -> str:
        return f"yolo ({self._model_path})"
