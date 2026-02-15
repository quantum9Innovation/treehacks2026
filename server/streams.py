"""WebRTC video stream tracks using aiortc."""

import logging
import time

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from .hardware import HardwareManager

logger = logging.getLogger("server.streams")

# Singleton relay for fanning out tracks to multiple peer connections
_relay: MediaRelay | None = None


def get_relay() -> MediaRelay:
    global _relay
    if _relay is None:
        _relay = MediaRelay()
    return _relay


class CameraVideoTrack(VideoStreamTrack):
    """Reads latest frame from HardwareManager, emits as WebRTC video."""

    kind = "video"

    def __init__(self, hw: HardwareManager, stream: str = "color"):
        super().__init__()
        self._hw = hw
        self._stream = stream  # "color" or "depth"
        self._start = time.time()

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        bundle = self._hw.latest_frame
        if bundle is None:
            # No frame yet — send black
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = VideoFrame.from_ndarray(black, format="rgb24")
        else:
            if self._stream == "color":
                bgr = bundle.color_image
            else:
                bgr = bundle.depth_image

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = VideoFrame.from_ndarray(rgb, format="rgb24")

        frame.pts = pts
        frame.time_base = time_base
        return frame
