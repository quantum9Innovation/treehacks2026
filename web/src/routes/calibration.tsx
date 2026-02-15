import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useRef, useState } from 'react'
import { CameraFeed } from '@/components/camera/camera-feed'
import { CalibrationWizard } from '@/components/calibration/calibration-wizard'
import { useWebSocket } from '@/lib/hooks/use-websocket'
import { useRobot } from '@/lib/robot-context'
import type { WSMessage } from '@/lib/types'

export const Route = createFileRoute('/calibration')({
  component: CalibrationPage,
})

function CalibrationPage() {
  const [lastCalMsg, setLastCalMsg] = useState<WSMessage | null>(null)
  const clickHandlerRef = useRef<((px: number, py: number) => void) | null>(null)
  const { dispatch } = useRobot()

  const onWsMessage = useCallback((msg: WSMessage) => {
    if (msg.type === 'calibration.progress' || msg.type === 'calibration.result') {
      setLastCalMsg(msg)
    }
  }, [])

  useWebSocket(onWsMessage)

  const setClickHandler = useCallback(
    (handler: ((px: number, py: number) => void) | null) => {
      clickHandlerRef.current = handler
      dispatch({
        type: 'SET_INTERACTION_MODE',
        payload: handler ? 'segment' : 'off',
      })
    },
    [dispatch],
  )

  const handleClickInfo = useCallback(
    (info: { pixel_x: number; pixel_y: number }) => {
      if (clickHandlerRef.current) {
        clickHandlerRef.current(info.pixel_x, info.pixel_y)
      }
    },
    [],
  )

  return (
    <div className="flex h-full gap-4 p-4">
      {/* Camera panel */}
      <div className="flex-[3] min-w-0">
        <CameraFeed onClickInfo={handleClickInfo} showToolbar={false} />
      </div>

      {/* Calibration panel */}
      <div className="w-80">
        <CalibrationWizard onClickPixel={setClickHandler} lastWsMessage={lastCalMsg} />
      </div>
    </div>
  )
}
