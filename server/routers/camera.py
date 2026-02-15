"""Camera endpoints: WebRTC signaling and snapshot fallback."""

import base64
import logging

import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..hardware import HardwareManager
from ..streams import CameraVideoTrack, get_relay

logger = logging.getLogger("server.routers.camera")
router = APIRouter(prefix="/api/camera", tags=["camera"])

# Track active peer connections for cleanup
_pcs: set[RTCPeerConnection] = set()


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


class WebRTCAnswer(BaseModel):
    sdp: str
    type: str


class SnapshotResponse(BaseModel):
    color_b64: str
    depth_b64: str
    width: int
    height: int


@router.post("/webrtc/offer", response_model=WebRTCAnswer)
async def webrtc_offer(body: WebRTCOffer, request: Request):
    """WebRTC signaling: receive SDP offer, return SDP answer."""
    hw: HardwareManager = request.app.state.hardware
    relay = get_relay()

    offer = RTCSessionDescription(sdp=body.sdp, type=body.type)
    pc = RTCPeerConnection()
    _pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        logger.info(f"WebRTC connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            _pcs.discard(pc)

    # Add color and depth video tracks
    color_track = CameraVideoTrack(hw, stream="color")
    depth_track = CameraVideoTrack(hw, stream="depth")
    pc.addTrack(relay.subscribe(color_track))
    pc.addTrack(relay.subscribe(depth_track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return WebRTCAnswer(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
    )


@router.get("/snapshot", response_model=SnapshotResponse)
async def snapshot(request: Request):
    """Grab a single JPEG snapshot (fallback if WebRTC unavailable)."""
    hw: HardwareManager = request.app.state.hardware
    bundle = hw.latest_frame
    if bundle is None:
        raise HTTPException(status_code=503, detail="Camera not ready")

    _, color_jpg = cv2.imencode(
        ".jpg", bundle.color_image, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    _, depth_jpg = cv2.imencode(
        ".jpg", bundle.depth_image, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    return SnapshotResponse(
        color_b64=base64.b64encode(color_jpg.tobytes()).decode(),
        depth_b64=base64.b64encode(depth_jpg.tobytes()).decode(),
        width=640,
        height=480,
    )
