import { createFileRoute } from '@tanstack/react-router'
import { useEffect } from 'react'
import { useRobot } from '@/lib/robot-context'
import { useWebRTCStream } from '@/lib/hooks/use-webrtc-stream'
import { useWebSocket } from '@/lib/hooks/use-websocket'
import { VideoPanel } from '@/components/display/video-panel'

export const Route = createFileRoute('/display')({
  component: DisplayPage,
})

function DisplayPage() {
  const { dispatch } = useRobot()
  const { connected, tracks } = useWebRTCStream()
  useWebSocket()

  useEffect(() => {
    dispatch({ type: 'SET_CAMERA_CONNECTED', payload: connected })
  }, [connected, dispatch])

  return (
    <div className="grid h-full grid-cols-2 grid-rows-2 gap-4 p-4">
      <VideoPanel track={tracks[0] ?? null} label="Color" />
      <VideoPanel track={tracks[1] ?? null} label="Depth" />
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">Panel 3</span>
        <div className="flex-1 rounded-xl border" />
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-muted-foreground">Panel 4</span>
        <div className="flex-1 rounded-xl border" />
      </div>
    </div>
  )
}
