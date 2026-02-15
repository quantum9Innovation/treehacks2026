import { Badge } from '@/components/ui/badge'
import { useRobot } from '@/lib/robot-context'

function ConnectionDot({ connected }: { connected: boolean }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}
    />
  )
}

export function StatusBar() {
  const { state } = useRobot()

  const agentBadge = {
    idle: 'secondary',
    running: 'default',
    awaiting_confirm: 'destructive',
    error: 'destructive',
  } as const

  return (
    <header className="flex h-12 items-center justify-between border-b bg-background px-4">
      <div className="flex items-center gap-4">
        <h1 className="text-sm font-semibold">RoArm Control</h1>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <ConnectionDot connected={state.cameraConnected} />
            Camera
          </span>
          <span className="flex items-center gap-1">
            <ConnectionDot connected={state.armConnected} />
            Arm
          </span>
          <span className="flex items-center gap-1">
            <ConnectionDot connected={state.wsConnected} />
            WS
          </span>
        </div>
      </div>
      <div className="flex items-center gap-4 text-xs">
        {state.armPosition && (
          <span className="font-mono text-muted-foreground">
            X:{state.armPosition.x.toFixed(0)} Y:{state.armPosition.y.toFixed(0)} Z:
            {state.armPosition.z.toFixed(0)}
          </span>
        )}
        {state.calibrationLoaded && (
          <Badge variant="outline" className="text-xs">
            Cal {state.calibrationRMSE ? `${state.calibrationRMSE.toFixed(1)}mm` : 'OK'}
          </Badge>
        )}
        <Badge variant={agentBadge[state.agentState]} className="text-xs">
          {state.agentState}
        </Badge>
      </div>
    </header>
  )
}
