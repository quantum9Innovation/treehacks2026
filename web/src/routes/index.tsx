import { createFileRoute } from '@tanstack/react-router'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { CameraFeed } from '@/components/camera/camera-feed'
import { MovementGrid } from '@/components/control/movement-grid'
import { GripperControl } from '@/components/control/gripper-control'
import { PositionReadout } from '@/components/control/position-readout'
import { StepSizeSelector } from '@/components/control/step-size-selector'
import { ProbeGround } from '@/components/control/probe-ground'
import { ArmSelector } from '@/components/control/arm-selector'
import { useKeyboardControls } from '@/lib/hooks/use-keyboard-controls'
import { useWebSocket } from '@/lib/hooks/use-websocket'
import { useState } from 'react'

export const Route = createFileRoute('/')({
  component: ControlPage,
})

function ControlPage() {
  useWebSocket()
  useKeyboardControls(true)
  const [clickInfo, setClickInfo] = useState<{ pixel_x: number; pixel_y: number } | null>(null)
  const [segResult, setSegResult] = useState<{
    arm_coordinates: { x: number; y: number; z: number } | null
    score: number
  } | null>(null)

  return (
    <div className="flex h-full gap-2 p-2">
      {/* Camera panel */}
      <div className="flex-[2] min-w-0">
        <CameraFeed
          onClickInfo={setClickInfo}
          onSegmentResult={setSegResult}
        />
      </div>

      {/* Controls panel */}
      <div className="flex w-64 flex-col gap-1.5">
        <Card>
          <CardHeader className="py-1.5 px-2">
            <CardTitle className="text-xs">Arm</CardTitle>
          </CardHeader>
          <CardContent className="px-2 pb-2">
            <ArmSelector />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="py-1.5 px-2">
            <CardTitle className="text-xs">Position</CardTitle>
          </CardHeader>
          <CardContent className="px-2 pb-2">
            <PositionReadout />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="py-1.5 px-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs">Movement</CardTitle>
              <StepSizeSelector />
            </div>
          </CardHeader>
          <CardContent className="px-2 pb-2">
            <MovementGrid />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="py-1.5 px-2">
            <CardTitle className="text-xs">Gripper</CardTitle>
          </CardHeader>
          <CardContent className="px-2 pb-2">
            <GripperControl />
          </CardContent>
        </Card>

        <ProbeGround />

        {/* Debug info */}
        {(clickInfo || segResult) && (
          <Card>
            <CardHeader className="py-1.5 px-2">
              <CardTitle className="text-xs">Debug</CardTitle>
            </CardHeader>
            <CardContent className="px-2 pb-2 text-xs font-mono space-y-0.5">
              {clickInfo && (
                <p>
                  Click: ({clickInfo.pixel_x}, {clickInfo.pixel_y})
                </p>
              )}
              {segResult && (
                <>
                  <p>Score: {segResult.score.toFixed(3)}</p>
                  {segResult.arm_coordinates && (
                    <p>
                      Arm: ({segResult.arm_coordinates.x.toFixed(0)},{' '}
                      {segResult.arm_coordinates.y.toFixed(0)},{' '}
                      {segResult.arm_coordinates.z.toFixed(0)})
                    </p>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
