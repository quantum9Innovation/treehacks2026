"""Abstract vision backend interface."""

from abc import ABC, abstractmethod

import numpy as np

from ..models import DetectedObject


class VisionBackend(ABC):
    """Base class for object detection backends."""

    @abstractmethod
    def detect(
        self,
        color_image: np.ndarray,
        confidence_threshold: float = 0.3,
        classes: list[str] | None = None,
    ) -> list[DetectedObject]:
        """Detect objects in a color image.

        Args:
            color_image: BGR image as numpy array
            confidence_threshold: Minimum confidence to include
            classes: Optional filter list of class names

        Returns:
            List of DetectedObject with object_id, label, confidence,
            bbox, and center_px filled. Depth/3D fields are None.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the backend name for logging."""
        ...
