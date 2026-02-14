"""Data models for the object-aware agent pipeline."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BBox:
    """Pixel-space bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        """Center pixel (cx, cy)."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class Point3D:
    """3D point in millimeters."""

    x: float
    y: float
    z: float

    def to_dict(self) -> dict:
        return {"x": round(self.x, 1), "y": round(self.y, 1), "z": round(self.z, 1)}

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point3D":
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class DetectedObject:
    """A detected object carrying data through the full pipeline.

    Fields are filled progressively:
    - Vision backend fills: object_id, label, confidence, bbox, center_px, mask
    - Depth enrichment fills: depth_mm
    - Coordinate transform fills: position_cam, position_arm
    """

    object_id: int
    label: str
    confidence: float
    bbox: BBox
    center_px: tuple[int, int]
    mask: np.ndarray | None = field(default=None, repr=False)
    depth_mm: float | None = None
    position_cam: Point3D | None = None
    position_arm: Point3D | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict for LLM consumption (no numpy)."""
        d: dict = {
            "object_id": self.object_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
            "center_px": list(self.center_px),
        }
        if self.depth_mm is not None:
            d["depth_mm"] = round(self.depth_mm, 1)
        if self.position_cam is not None:
            d["position_cam_mm"] = self.position_cam.to_dict()
        if self.position_arm is not None:
            d["position_arm_mm"] = self.position_arm.to_dict()
        return d


@dataclass
class SceneState:
    """Full scene detection result."""

    objects: list[DetectedObject]
    image_width: int
    image_height: int

    def get_object(self, object_id: int) -> DetectedObject | None:
        """Look up object by ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def to_dict(self) -> dict:
        return {
            "num_objects": len(self.objects),
            "image_size": [self.image_width, self.image_height],
            "objects": [obj.to_dict() for obj in self.objects],
        }
