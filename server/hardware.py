"""Singleton hardware manager for camera, arm, vision, and coordinate transform."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("server.hardware")

# Single-thread executor for serializing hardware access
_hw_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hw")
_vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")


@dataclass
class FrameBundle:
    """A captured set of aligned frames."""

    color_image: np.ndarray  # BGR (H, W, 3)
    depth_image: np.ndarray  # Colorized depth (H, W, 3)
    depth_frame: object  # rs.depth_frame (typed as object to avoid import at module level)
    timestamp: float


class HardwareManager:
    """Owns all physical hardware as singletons with thread-safe access."""

    def __init__(
        self,
        arm_devices: list[str] | None = None,
        sam2_model: str = "tiny",
        sam2_device: str = "auto",
        calibration_path: str = "agent_v2/calibration_data.json",
        mock: bool = False,
        enable_sam2: bool = True,
        enable_gemini_vision: bool = True,
        google_api_key: str = "",
    ):
        self._arm_devices = arm_devices or [
            "/dev/ARM0", "/dev/ARM1", "/dev/ARM2", "/dev/ARM3"
        ]
        self._sam2_model = sam2_model
        self._sam2_device = sam2_device
        self._calibration_path = calibration_path
        self._mock = mock
        self._enable_sam2 = enable_sam2
        self._enable_gemini_vision = enable_gemini_vision
        self._google_api_key = google_api_key

        self.camera = None
        self.motion = None
        self.sam2 = None
        self.ct = None
        self.gemini_vision = None
        self.active_arm_device: str | None = None

        self._latest_frame: FrameBundle | None = None
        self._capture_task: asyncio.Task | None = None
        self._started = False

        # Lock for arm commands (serialize movements)
        self.arm_lock = asyncio.Lock()

        # Vision frame cache (set by look, used by segment)
        self.vision_color: np.ndarray | None = None
        self.vision_depth_frame = None

    @property
    def arm_devices(self) -> list[str]:
        return list(self._arm_devices)

    def clear_vision_cache(self):
        """Release cached vision frames to free memory."""
        self.vision_color = None
        self.vision_depth_frame = None

    async def start(self):
        """Initialize all hardware (camera, vision). Arm is connected separately via connect_arm."""
        loop = asyncio.get_event_loop()

        if self._mock:
            logger.info("Running in mock hardware mode")
            self._started = True
            return

        await loop.run_in_executor(_hw_executor, self._init_hardware)
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._started = True

    def _init_hardware(self):
        """Initialize camera and vision (runs in thread). Arm is deferred."""
        from pathlib import Path

        from agent.camera import RealSenseCamera
        from agent_v2.coordinate_transform import CoordinateTransform

        logger.info("Initializing camera...")
        self.camera = RealSenseCamera()
        self.camera.start()

        if self._enable_sam2:
            from agent_v2.vision.sam2_backend import SAM2Backend
            logger.info(f"Loading SAM2 ({self._sam2_model})...")
            self.sam2 = SAM2Backend(
                model_size=self._sam2_model, device=self._sam2_device
            )
        else:
            logger.info("SAM2 disabled")

        if self._enable_gemini_vision and self._google_api_key:
            from google import genai
            from agent_v3.gemini_vision import GeminiVision
            logger.info("Initializing Gemini ER vision...")
            vision_client = genai.Client(api_key=self._google_api_key)
            self.gemini_vision = GeminiVision(client=vision_client)
        else:
            logger.info("Gemini Vision disabled")

        logger.info("Initializing coordinate transform...")
        self.ct = CoordinateTransform(
            calibration_path=Path(self._calibration_path)
        )
        self.ct.set_intrinsics_from_camera(self.camera)

        logger.info("Hardware initialization complete (no arm connected yet)")

    def _connect_arm_sync(self, device: str):
        """Connect to an arm device (runs in thread)."""
        from motion_controller.motion import Motion

        # Disconnect existing arm if any
        if self.motion is not None:
            try:
                self.motion.home()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error homing previous arm: {e}")
            self.motion = None
            self.active_arm_device = None

        logger.info(f"Connecting to arm on {device}...")
        self.motion = Motion(port=device, inverted=True)
        self.active_arm_device = device

        logger.info("Moving to home (skipping ground probe — use control panel)...")
        self.motion.arm.move_init()
        time.sleep(1)

        logger.info(f"Arm connected on {device}")

    async def connect_arm(self, device: str):
        """Connect to a specific arm device."""
        async with self.arm_lock:
            await self.run_in_hw_thread(self._connect_arm_sync, device)

    def _disconnect_arm_sync(self):
        """Disconnect the current arm (runs in thread)."""
        if self.motion is not None:
            try:
                self.motion.home()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error homing arm: {e}")
            self.motion = None
            self.active_arm_device = None
            logger.info("Arm disconnected")

    async def disconnect_arm(self):
        """Disconnect the current arm."""
        async with self.arm_lock:
            await self.run_in_hw_thread(self._disconnect_arm_sync)

    async def _capture_loop(self):
        """Continuously capture frames at ~30fps."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                bundle = await loop.run_in_executor(
                    _hw_executor, self._capture_once
                )
                if bundle is not None:
                    self._latest_frame = bundle
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
            await asyncio.sleep(1 / 30)

    def _capture_once(self) -> FrameBundle | None:
        """Capture one aligned frame set (runs in thread)."""
        if self.camera is None or self.ct is None:
            return None

        try:
            frames = self.camera.pipeline.wait_for_frames()
            aligned = self.ct.get_aligned_frames(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            color_image = np.asanyarray(color_frame.get_data())
            colorized = self.camera.colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(colorized.get_data())

            return FrameBundle(
                color_image=color_image,
                depth_image=depth_image,
                depth_frame=depth_frame,
                timestamp=time.time(),
            )
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None

    @property
    def latest_frame(self) -> FrameBundle | None:
        return self._latest_frame

    @property
    def is_ready(self) -> bool:
        return self._started

    @property
    def has_calibration(self) -> bool:
        if self.ct is None:
            return False
        return self.ct._M is not None

    async def run_in_hw_thread(self, fn, *args):
        """Run a function in the hardware thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_hw_executor, fn, *args)

    async def run_in_vision_thread(self, fn, *args):
        """Run a function in the vision thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_vision_executor, fn, *args)

    async def shutdown(self):
        """Clean up hardware."""
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass

        if self._mock:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_hw_executor, self._shutdown_sync)

        # Shut down thread pools so their non-daemon threads don't block exit
        _hw_executor.shutdown(wait=False)
        _vision_executor.shutdown(wait=False)

    def _shutdown_sync(self):
        if self.motion:
            try:
                self.motion.home()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error homing arm: {e}")
        if self.camera:
            try:
                self.camera.stop()
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
