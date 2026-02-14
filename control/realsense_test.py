import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template

app = Flask(__name__)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Colorizer for depth visualization
colorizer = rs.colorizer()


def generate_color_frames():
    """Generator for color stream MJPEG frames."""
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        _, jpeg = cv2.imencode('.jpg', color_image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


def generate_depth_frames():
    """Generator for depth stream MJPEG frames (colorized)."""
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Apply colormap to depth frame
        colorized_depth = colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized_depth.get_data())
        _, jpeg = cv2.imencode('.jpg', depth_image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/video_feed/color')
def video_feed_color():
    """MJPEG stream for color frames."""
    return Response(generate_color_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/depth')
def video_feed_depth():
    """MJPEG stream for depth frames."""
    return Response(generate_depth_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        pipeline.stop()
