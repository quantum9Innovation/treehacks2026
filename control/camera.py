"""
sender_cam_stream.py
- Probes working camera indices on Windows
- You choose one
- Serves MJPEG at http://<sender_ip>:5000/mjpeg
- Broadcasts discovery packets on LAN so receiver auto-finds it

Install:
  pip install opencv-python flask

Run:
  python sender_cam_stream.py
"""

import json
import socket
import threading
import time
from typing import List, Optional, Tuple

import cv2
from flask import Flask, Response, jsonify

# -----------------------
# Config
# -----------------------
HTTP_PORT = 5000
DISCOVERY_PORT = 37020
DISCOVERY_MAGIC = "LAN_CAM_V1"
BROADCAST_INTERVAL_SEC = 1.0

DEFAULT_MAX_CAM_INDEX = 10

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)

latest_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()
stop_flag = False


def get_lan_ip() -> str:
    """Best-effort pick the LAN IP used for outbound traffic."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def probe_cameras(max_index: int = DEFAULT_MAX_CAM_INDEX) -> List[int]:
    """
    Return indices that open and produce a frame.
    CAP_DSHOW tends to be best on Windows.
    """
    working = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            working.append(idx)
    return working


def detect_max_resolution(camera_index: int) -> Tuple[int, int]:
    """Open camera, request a huge resolution, read back what the driver actually set."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return (1920, 1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w if w > 0 else 1920, h if h > 0 else 1080)


def capture_loop(
    camera_index: int, width: int, height: int, fps: int, jpeg_quality: int
) -> None:
    global latest_jpeg, stop_flag

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    frame_interval = 1.0 / max(1, fps)

    try:
        while not stop_flag:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                jpg = buf.tobytes()
                with jpeg_lock:
                    latest_jpeg = jpg

            dt = time.time() - t0
            sleep_for = frame_interval - dt
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()


def mjpeg_generator():
    boundary = b"--frame"
    while True:
        with jpeg_lock:
            jpg = latest_jpeg
        if jpg is None:
            time.sleep(0.02)
            continue

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(jpg)}\r\n\r\n".encode()
        yield jpg + b"\r\n"


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/mjpeg")
def mjpeg():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


def discovery_broadcast_loop(sender_ip: str, http_port: int) -> None:
    """
    Broadcasts a small UDP packet so receivers on the LAN can auto-discover the sender.
    """
    msg = {
        "magic": DISCOVERY_MAGIC,
        "ip": sender_ip,
        "port": http_port,
        "path": "/mjpeg",
        "ts": time.time(),
    }
    payload = json.dumps(msg).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    broadcast_addr = ("255.255.255.255", DISCOVERY_PORT)

    try:
        while not stop_flag:
            try:
                sock.sendto(payload, broadcast_addr)
            except Exception:
                pass
            time.sleep(BROADCAST_INTERVAL_SEC)
    finally:
        sock.close()


def main():
    print("\n--- Sender: Webcam LAN Stream ---")
    cams = probe_cameras(DEFAULT_MAX_CAM_INDEX)
    if not cams:
        print(f"No working cameras found in indices 0..{DEFAULT_MAX_CAM_INDEX - 1}.")
        print("If needed, increase DEFAULT_MAX_CAM_INDEX in the file.")
        return

    print("Working camera indices:", cams)
    while True:
        s = input("Choose camera index: ").strip()
        if s.isdigit() and int(s) in cams:
            cam_index = int(s)
            break
        print("Invalid choice. Pick one of:", cams)

    max_w, max_h = detect_max_resolution(cam_index)
    print(f"Detected max resolution: {max_w}x{max_h}")

    width = int(input(f"Width (default {max_w}): ").strip() or str(max_w))
    height = int(input(f"Height (default {max_h}): ").strip() or str(max_h))
    fps = int(input("FPS (default 120): ").strip() or "120")
    jpeg_quality = int(input("JPEG quality 1-100 (default 100): ").strip() or "100")

    sender_ip = get_lan_ip()

    # Start capture thread
    t_cap = threading.Thread(
        target=capture_loop,
        args=(cam_index, width, height, fps, jpeg_quality),
        daemon=True,
    )
    t_cap.start()

    # Start discovery broadcast thread
    t_discovery = threading.Thread(
        target=discovery_broadcast_loop,
        args=(sender_ip, HTTP_PORT),
        daemon=True,
    )
    t_discovery.start()

    print("\nSender ready.")
    print(f"Auto-discovery: UDP broadcast on port {DISCOVERY_PORT}")
    print(f"MJPEG endpoint (FYI): http://{sender_ip}:{HTTP_PORT}/mjpeg")
    print(
        "If receiver can’t find it, check Windows Firewall for UDP 37020 and TCP 5000.\n"
    )

    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)


if __name__ == "__main__":
    main()
