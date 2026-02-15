"""J5 webcam MJPEG stream + LAN UDP discovery."""

import json
import logging
import socket
import threading
import time
from typing import Optional

import cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger("server.routers.webcam")
router = APIRouter(prefix="/api/webcam", tags=["webcam"])

# ---------------------------------------------------------------------------
# Module-level state (started/stopped via lifespan helpers)
# ---------------------------------------------------------------------------
_latest_jpeg: Optional[bytes] = None
_frame_ready = threading.Condition()
_stop_flag = False
_capture_thread: Optional[threading.Thread] = None
_discovery_thread: Optional[threading.Thread] = None

DISCOVERY_PORT = 37020
DISCOVERY_MAGIC = "LAN_CAM_V1"
BROADCAST_INTERVAL_SEC = 1.0


def _get_lan_ip() -> str:
    """Best-effort LAN IP used for outbound traffic."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------
def _capture_loop(
    device: int | str,
    width: int,
    height: int,
    fps: int,
    jpeg_quality: int,
) -> None:
    global _latest_jpeg, _stop_flag

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.error("Could not open webcam device %s", device)
        return

    # Request MJPEG from the camera so the driver delivers pre-compressed
    # frames instead of raw YUYV that OpenCV must software-convert.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    logger.info(
        "Webcam opened: device=%s resolution=%dx%d fps=%.1f fourcc=%s",
        device, actual_w, actual_h, actual_fps, fourcc_str,
    )

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    try:
        while not _stop_flag:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                jpg = buf.tobytes()
                with _frame_ready:
                    _latest_jpeg = jpg
                    _frame_ready.notify_all()
    finally:
        cap.release()
        logger.info("Webcam capture stopped")


# ---------------------------------------------------------------------------
# UDP discovery broadcast
# ---------------------------------------------------------------------------
def _discovery_loop(sender_ip: str, http_port: int) -> None:
    msg = json.dumps({
        "magic": DISCOVERY_MAGIC,
        "ip": sender_ip,
        "port": http_port,
        "path": "/api/webcam/mjpeg",
        "ts": time.time(),
    }).encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    dest = ("255.255.255.255", DISCOVERY_PORT)

    try:
        while not _stop_flag:
            try:
                sock.sendto(msg, dest)
            except Exception:
                pass
            time.sleep(BROADCAST_INTERVAL_SEC)
    finally:
        sock.close()
        logger.info("Discovery broadcast stopped")


# ---------------------------------------------------------------------------
# Lifespan helpers (called from app.py)
# ---------------------------------------------------------------------------
def start_webcam(
    device: int | str = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    jpeg_quality: int = 85,
    http_port: int = 8420,
) -> None:
    """Start capture + discovery threads."""
    global _stop_flag, _capture_thread, _discovery_thread
    _stop_flag = False

    _capture_thread = threading.Thread(
        target=_capture_loop,
        args=(device, width, height, fps, jpeg_quality),
        daemon=True,
    )
    _capture_thread.start()

    sender_ip = _get_lan_ip()
    _discovery_thread = threading.Thread(
        target=_discovery_loop,
        args=(sender_ip, http_port),
        daemon=True,
    )
    _discovery_thread.start()

    logger.info(
        "Webcam streaming started: device=%s, MJPEG at http://%s:%d/api/webcam/mjpeg",
        device, sender_ip, http_port,
    )


def stop_webcam() -> None:
    """Signal threads to stop and wake any waiting generators."""
    global _stop_flag
    _stop_flag = True
    with _frame_ready:
        _frame_ready.notify_all()


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------
def _mjpeg_generator():
    while not _stop_flag:
        with _frame_ready:
            _frame_ready.wait(timeout=1.0)
            jpg = _latest_jpeg
        if jpg is None:
            continue

        # Yield the entire frame as one chunk to avoid per-yield flush overhead.
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
            + jpg + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/mjpeg")
async def mjpeg_stream():
    """Multipart MJPEG stream of the J5 webcam."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@router.get("/health")
async def webcam_health():
    """Quick check that the webcam capture is producing frames."""
    with _frame_ready:
        has_frame = _latest_jpeg is not None
    return {"ok": has_frame}
