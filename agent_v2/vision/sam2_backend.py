"""SAM2 (Segment Anything Model 2) backend for point-prompt segmentation."""

import logging
import os
import time

import numpy as np

logger = logging.getLogger("agent_v2.sam2")


class SAM2Backend:
    """Wraps SAM2ImagePredictor for point-prompt segmentation.

    Workflow:
        1. set_image(color_bgr) — encode image (~2-5s on CPU, run once per look())
        2. segment_point(px, py) — fast point query (~0.3s, reuses features)
    """

    def __init__(self, model_size: str = "tiny", device: str = "auto"):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model_map = {
            "tiny": "facebook/sam2.1-hiera-tiny",
            "small": "facebook/sam2.1-hiera-small",
            "base_plus": "facebook/sam2.1-hiera-base-plus",
            "large": "facebook/sam2.1-hiera-large",
        }
        model_id = model_map.get(model_size)
        if model_id is None:
            raise ValueError(
                f"Unknown SAM2 model size: {model_size!r}. Choose from {list(model_map)}"
            )

        resolved_device = self._resolve_device(torch, device)
        self.device = resolved_device

        logger.info(f"Loading SAM2 model: {model_id} (device={resolved_device})")
        t0 = time.time()

        self.predictor = SAM2ImagePredictor.from_pretrained(
            model_id, device=torch.device(resolved_device)
        )
        self._torch = torch

        logger.info(f"SAM2 model loaded in {time.time() - t0:.1f}s")
        self._image_set = False

    def name(self) -> str:
        return "sam2"

    @staticmethod
    def _resolve_device(torch, requested: str) -> str:
        # Precedence:
        # 1) explicit constructor arg (unless "auto")
        # 2) env override (SAM2_DEVICE or SAM_DEVICE)
        # 3) auto: cuda -> mps -> cpu
        req = (requested or "auto").strip()
        if req.lower() != "auto":
            return req

        env_dev = (
            os.environ.get("SAM2_DEVICE") or os.environ.get("SAM_DEVICE") or ""
        ).strip()
        if env_dev:
            return env_dev

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda"
        if (
            getattr(torch, "backends", None) is not None
            and getattr(torch.backends, "mps", None) is not None
        ):
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def set_image(self, color_bgr: np.ndarray) -> float:
        """Encode an image for subsequent point queries.

        Args:
            color_bgr: BGR image from OpenCV (H, W, 3)

        Returns:
            Encoding time in seconds.
        """
        import cv2

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        with self._torch.inference_mode():
            self.predictor.set_image(color_rgb)
        elapsed = time.time() - t0
        self._image_set = True
        logger.info(f"SAM2 image encoded in {elapsed:.2f}s")
        return elapsed

    def segment_point(
        self, pixel_x: int, pixel_y: int
    ) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
        """Run point-prompt segmentation at the given pixel.

        Args:
            pixel_x: X coordinate in the image
            pixel_y: Y coordinate in the image

        Returns:
            mask: Boolean mask (H, W)
            score: Confidence score (0-1)
            bbox: (x1, y1, x2, y2) bounding box of the mask
        """
        if not self._image_set:
            raise RuntimeError("Call set_image() before segment_point()")

        point_coords = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

        with self._torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # Pick the mask with the highest score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(bool)
        score = float(scores[best_idx])

        # Compute bounding box from mask
        bbox = self._mask_bbox(mask)

        return mask, score, bbox

    @staticmethod
    def mask_centroid(mask: np.ndarray) -> tuple[int, int]:
        """Compute the centroid of a boolean mask.

        Args:
            mask: Boolean mask (H, W)

        Returns:
            (cx, cy) pixel coordinates of the centroid
        """
        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise ValueError("Empty mask — cannot compute centroid")
        return int(np.mean(xs)), int(np.mean(ys))

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        """Compute bounding box (x1, y1, x2, y2) from a boolean mask."""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
