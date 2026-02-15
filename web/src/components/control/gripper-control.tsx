import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { useRobot } from '@/lib/robot-context'
import { armApi } from '@/lib/api'

export function GripperControl() {
  const { state } = useRobot()
  const [localAngle, setLocalAngle] = useState(state.gripperAngle)
  const [dragging, setDragging] = useState(false)

  // Sync from server (via WebSocket) when not dragging
  useEffect(() => {
    if (!dragging) setLocalAngle(state.gripperAngle)
  }, [state.gripperAngle, dragging])

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">Gripper</span>
        <span className="text-xs font-mono">{localAngle.toFixed(0)}°</span>
      </div>
      <Slider
        value={[localAngle]}
        onValueChange={([v]) => {
          setDragging(true)
          setLocalAngle(v)
        }}
        onValueCommit={([v]) => {
          setDragging(false)
          armApi.gripper(v)
        }}
        min={0}
        max={90}
        step={5}
      />
      <div className="flex gap-1">
        <Button variant="outline" size="sm" className="flex-1 text-xs" onClick={() => armApi.gripper(0)}>
          Close [F]
        </Button>
        <Button variant="outline" size="sm" className="flex-1 text-xs" onClick={() => armApi.gripper(90)}>
          Open [R]
        </Button>
      </div>
    </div>
  )
}
