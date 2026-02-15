import { useCallback, useRef, useState } from 'react'
import { useRobot } from '@/lib/robot-context'
import { useWebRTCStream } from '@/lib/hooks/use-webrtc-stream'
import { visionApi, armApi } from '@/lib/api'
import { CAMERA_WIDTH, CAMERA_HEIGHT } from '@/lib/constants'
import { GridOverlay } from './grid-overlay'
import { CameraToolbar } from './camera-toolbar'

interface CameraFeedProps {
  onSegmentResult?: (result: {
    annotated_b64: string
    arm_coordinates: { x: number; y: number; z: number } | null
    score: number
  }) => void
  onClickInfo?: (info: { pixel_x: number; pixel_y: number }) => void
  showToolbar?: boolean
}

export function CameraFeed({ onSegmentResult, onClickInfo, showToolbar = true }: CameraFeedProps) {
  const { state } = useRobot()
  const { videoRef, connected } = useWebRTCStream()
  const containerRef = useRef<HTMLDivElement>(null)
  const [segmentOverlay, setSegmentOverlay] = useState<string | null>(null)
  const [showGrid, setShowGrid] = useState(true)

  const handleClick = useCallback(
    async (e: React.MouseEvent<HTMLDivElement>) => {
      if (state.interactionMode === 'off') return
      if (!containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const scaleX = CAMERA_WIDTH / rect.width
      const scaleY = CAMERA_HEIGHT / rect.height
      const pixel_x = Math.round((e.clientX - rect.left) * scaleX)
      const pixel_y = Math.round((e.clientY - rect.top) * scaleY)

      if (pixel_x < 0 || pixel_x >= CAMERA_WIDTH || pixel_y < 0 || pixel_y >= CAMERA_HEIGHT) return

      onClickInfo?.({ pixel_x, pixel_y })

      try {
        if (state.interactionMode === 'segment') {
          const result = await visionApi.segment(pixel_x, pixel_y)
          setSegmentOverlay(`data:image/jpeg;base64,${result.annotated_b64}`)
          onSegmentResult?.({
            annotated_b64: result.annotated_b64,
            arm_coordinates: result.arm_coordinates,
            score: result.score,
          })
        } else if (state.interactionMode === 'touch') {
          await armApi.gotoPixel(pixel_x, pixel_y)
        }
      } catch (err) {
        console.error('Click action failed:', err)
      }
    },
    [state.interactionMode, onSegmentResult, onClickInfo],
  )

  return (
    <div className="flex flex-col gap-2">
      {showToolbar && (
        <CameraToolbar
          showGrid={showGrid}
          onToggleGrid={() => setShowGrid(!showGrid)}
          onClearOverlay={() => setSegmentOverlay(null)}
        />
      )}
      <div
        ref={containerRef}
        className="relative aspect-[4/3] overflow-hidden rounded-lg border bg-black"
        onClick={handleClick}
        style={{
          cursor: state.interactionMode !== 'off' ? 'crosshair' : 'default',
        }}
      >
        {/* WebRTC video */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 h-full w-full object-contain"
        />

        {/* Segmentation overlay */}
        {segmentOverlay && (
          <img
            src={segmentOverlay}
            alt="Segmentation"
            className="absolute inset-0 h-full w-full object-contain opacity-80"
          />
        )}

        {/* Grid overlay */}
        {showGrid && (
          <svg
            className="absolute inset-0 h-full w-full pointer-events-none"
            viewBox={`0 0 ${CAMERA_WIDTH} ${CAMERA_HEIGHT}`}
            preserveAspectRatio="xMidYMid meet"
          >
            <GridOverlay />
          </svg>
        )}

        {/* Connection status overlay */}
        {!connected && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <p className="text-sm text-white">Connecting to camera...</p>
          </div>
        )}
      </div>
    </div>
  )
}
