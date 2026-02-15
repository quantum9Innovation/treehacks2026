"""RealSense camera wrapper for capturing color and depth frames."""

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """RealSense camera wrapper for capturing color and depth frames."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        min_depth_m: float = 0.2,
        max_depth_m: float = 1.5,
    ):
        """
        Initialize the RealSense camera.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            min_depth_m: Minimum depth for colorization in meters
            max_depth_m: Maximum depth for colorization in meters
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.colorizer = rs.colorizer()
        self._configure_colorizer()
        self._started = False
        self._configure()

    def _configure_colorizer(self) -> None:
        """Configure the depth colorizer with fixed range (red=close, blue=far)."""
        # Use fixed preset (1) instead of dynamic histogram equalization
        self.colorizer.set_option(rs.option.visual_preset, 1)  # 1 = Fixed
        self.colorizer.set_option(rs.option.min_distance, self.min_depth_m)
        self.colorizer.set_option(rs.option.max_distance, self.max_depth_m)

    def _configure(self) -> None:
        """Configure the RealSense pipeline."""
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )

    def start(self) -> None:
        """Start the camera pipeline."""
        if not self._started:
            print("Starting RealSense camera...")
            self.pipeline.start(self.config)
            self._started = True
            print("Camera started.")
            # # Save debug image on startup
            # self._save_debug_image()

    def stop(self) -> None:
        """Stop the camera pipeline."""
        if self._started:
            self.pipeline.stop()
            self._started = False
            print("Camera stopped.")

    def _save_debug_image(self) -> None:
        """Save a debug image on startup."""
        import time

        # Wait a moment for camera to stabilize
        time.sleep(0.5)
        try:
            color_image, depth_image = self.capture_frames()
            debug_path = "/tmp/vlm_agent_debug.jpg"
            cv2.imwrite(debug_path, color_image)
            print(f"Debug image saved to: {debug_path}")
        except Exception as e:
            print(f"Failed to save debug image: {e}")

    def _draw_axes(self, image: np.ndarray) -> np.ndarray:
        """Draw coordinate axes overlay on image with tick marks."""
        h, w = image.shape[:2]
        center_x = w // 2
        center_y = h - h // 4  # Origin a quarter way up from bottom

        # Colors (BGR)
        x_color = (0, 0, 255)  # Red for X axis
        y_color = (0, 255, 0)  # Green for Y axis
        tick_color = (255, 255, 255)  # White for ticks

        # Y axis length in pixels (from center to right edge)
        y_axis_length = w - 30 - center_x
        # Y axis represents ~800mm, so pixels per 100mm
        y_pixels_per_100mm = y_axis_length / 8.0

        # X axis goes halfway up from origin (represents ~400mm)
        x_axis_length = (h - 80) // 2  # Half of full height
        x_end_y = center_y - x_axis_length
        # X axis represents ~400mm, so pixels per 100mm
        x_pixels_per_100mm = x_axis_length / 4.0

        # Draw X axis (vertical line going up from origin - like 12 o'clock)
        cv2.line(image, (center_x, center_y), (center_x, x_end_y), x_color, 2)
        # X axis arrow head
        cv2.arrowedLine(
            image,
            (center_x, x_end_y + 20),
            (center_x, x_end_y),
            x_color,
            2,
            tipLength=0.5,
        )
        # X label
        cv2.putText(
            image,
            "X",
            (center_x + 10, x_end_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            x_color,
            2,
        )

        # Draw X axis ticks (every 100mm)
        for i in range(1, 5):  # 100, 200, 300, 400mm
            tick_y = int(center_y - i * x_pixels_per_100mm)
            cv2.line(
                image, (center_x - 5, tick_y), (center_x + 5, tick_y), tick_color, 1
            )
            cv2.putText(
                image,
                f"{i * 100}",
                (center_x + 8, tick_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                tick_color,
                1,
            )

        # Draw Y axis (horizontal line going right from bottom center - like 3 o'clock)
        cv2.line(image, (center_x, center_y), (w - 30, center_y), y_color, 2)
        # Y axis arrow head
        cv2.arrowedLine(
            image, (w - 50, center_y), (w - 30, center_y), y_color, 2, tipLength=0.5
        )
        # Y label
        cv2.putText(
            image,
            "Y",
            (w - 40, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            y_color,
            2,
        )

        # Draw Y axis ticks (every 100mm)
        for i in range(1, 9):  # 100, 200, ... 800mm
            tick_x = int(center_x + i * y_pixels_per_100mm)
            if tick_x < w - 40:  # Don't draw past arrow
                cv2.line(
                    image, (tick_x, center_y - 5), (tick_x, center_y + 5), tick_color, 1
                )
                cv2.putText(
                    image,
                    f"{i * 100}",
                    (tick_x - 10, center_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    tick_color,
                    1,
                )

        # Draw origin point
        cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(
            image,
            "0",
            (center_x - 15, center_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            tick_color,
            1,
        )

        return image

    def _draw_depth_scale(
        self, image: np.ndarray, min_depth_m: float, max_depth_m: float
    ) -> np.ndarray:
        """Draw a depth scale legend matching RealSense colorizer (red=close, blue=far)."""
        h, w = image.shape[:2]

        # Scale bar dimensions
        bar_width = 20
        bar_height = h - 60
        bar_x = w - 50
        bar_y = 30

        # Create a vertical gradient (0-255) and apply Jet colormap
        # RealSense: red=close, blue=far (inverted from standard Jet)
        gradient = (
            np.linspace(0, 255, bar_height).astype(np.uint8).reshape(bar_height, 1)
        )
        gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

        # Draw the gradient bar: top = close (red), bottom = far (blue)
        # Jet: 0=blue, 255=red, so use directly (top gets high values = red = close)
        for i in range(bar_height):
            color = gradient_color[
                bar_height - 1 - i, 0
            ]  # top=red(close), bottom=blue(far)
            cv2.line(
                image,
                (bar_x, bar_y + i),
                (bar_x + bar_width, bar_y + i),
                (int(color[0]), int(color[1]), int(color[2])),
                1,
            )

        # Draw border
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (255, 255, 255),
            1,
        )

        # Helper to draw outlined text (black outline, white fill)
        def draw_text(text: str, pos: tuple[int, int]) -> None:
            # Draw black outline
            cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            # Draw white text on top
            cv2.putText(
                image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

        # Add distance labels: top = close (min), bottom = far (max)
        draw_text(f"{min_depth_m:.2f}m", (bar_x - 52, bar_y + 12))
        mid_depth = (max_depth_m + min_depth_m) / 2
        draw_text(f"{mid_depth:.2f}m", (bar_x - 52, bar_y + bar_height // 2 + 4))
        draw_text(f"{max_depth_m:.2f}m", (bar_x - 52, bar_y + bar_height + 4))

        # Label
        draw_text("Depth", (bar_x - 5, bar_y - 10))

        return image

    def capture_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Capture color and depth frames.

        Returns:
            Tuple of (color_image, depth_image) as numpy arrays
        """
        if not self._started:
            self.start()

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture frames")

        color_image = np.asanyarray(color_frame.get_data())

        # Colorize depth for visualization (using fixed range from constructor)
        colorized_depth = self.colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized_depth.get_data())

        # Add depth scale to depth image (uses fixed min/max from colorizer config)
        depth_image = self._draw_depth_scale(
            depth_image.copy(), self.min_depth_m, self.max_depth_m
        )

        # Add coordinate axes overlay (color only)
        color_image = self._draw_axes(color_image.copy())

        return color_image, depth_image

    def capture_as_bytes(self, quality: int = 85) -> tuple[bytes, bytes]:
        """
        Capture frames and encode as JPEG bytes.

        Args:
            quality: JPEG quality (0-100)

        Returns:
            Tuple of (color_jpeg_bytes, depth_jpeg_bytes)
        """
        color_image, depth_image = self.capture_frames()

        _, color_jpeg = cv2.imencode(
            ".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        _, depth_jpeg = cv2.imencode(
            ".jpg", depth_image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        return color_jpeg.tobytes(), depth_jpeg.tobytes()

    def get_depth_at_pixel(self, x: int, y: int) -> float:
        """
        Get depth value at a specific pixel.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            Depth in millimeters
        """
        if not self._started:
            self.start()

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        return depth_frame.get_distance(x, y) * 1000  # Convert to mm

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
