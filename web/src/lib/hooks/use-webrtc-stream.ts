/** WebRTC hook for camera video streaming. */

import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE } from '../constants'

export function useWebRTCStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const [connected, setConnected] = useState(false)
  const [tracks, setTracks] = useState<MediaStreamTrack[]>([])
  const [activeTrackIndex, setActiveTrackIndex] = useState(0) // 0=color, 1=depth

  const connect = useCallback(async () => {
    try {
      const pc = new RTCPeerConnection({
        iceServers: [], // LAN only, no STUN needed
      })
      pcRef.current = pc

      let trackCount = 0
      pc.ontrack = (event) => {
        setTracks((prev) => [...prev, event.track])
        // Attach only the first track (color) by default
        if (trackCount === 0 && videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0]
        }
        trackCount++
      }

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'connected') {
          setConnected(true)
        } else if (['failed', 'closed', 'disconnected'].includes(pc.connectionState)) {
          setConnected(false)
          // Try reconnect after 3s
          setTimeout(connect, 3000)
        }
      }

      // Add transceivers for two video tracks (color + depth)
      pc.addTransceiver('video', { direction: 'recvonly' })
      pc.addTransceiver('video', { direction: 'recvonly' })

      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)

      const response = await fetch(`${API_BASE}/api/camera/webrtc/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
        }),
      })

      if (!response.ok) throw new Error('WebRTC signaling failed')

      const answer = await response.json()
      await pc.setRemoteDescription(new RTCSessionDescription(answer))
    } catch (err) {
      console.error('WebRTC connection failed:', err)
      setTimeout(connect, 3000)
    }
  }, [])

  const switchTrack = useCallback(
    (index: number) => {
      if (videoRef.current && tracks[index]) {
        const stream = new MediaStream([tracks[index]])
        videoRef.current.srcObject = stream
        setActiveTrackIndex(index)
      }
    },
    [tracks],
  )

  useEffect(() => {
    connect()
    return () => {
      pcRef.current?.close()
    }
  }, [connect])

  return { videoRef, connected, activeTrackIndex, switchTrack, tracks }
}
