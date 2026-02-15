import { Button } from '@/components/ui/button'
import { useRobot } from '@/lib/robot-context'
import { armApi } from '@/lib/api'
import { Home } from 'lucide-react'

export function PositionReadout() {
  const { state } = useRobot()
  const pos = state.armPosition

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2 font-mono text-sm">
        <div className="rounded bg-muted px-2 py-1 text-center">
          <div className="text-xs text-muted-foreground">X</div>
          <div>{pos ? pos.x.toFixed(1) : '---'}</div>
        </div>
        <div className="rounded bg-muted px-2 py-1 text-center">
          <div className="text-xs text-muted-foreground">Y</div>
          <div>{pos ? pos.y.toFixed(1) : '---'}</div>
        </div>
        <div className="rounded bg-muted px-2 py-1 text-center">
          <div className="text-xs text-muted-foreground">Z</div>
          <div>{pos ? pos.z.toFixed(1) : '---'}</div>
        </div>
      </div>
      <Button variant="outline" size="sm" className="w-full text-xs" onClick={() => armApi.home()}>
        <Home className="mr-1 h-3 w-3" />
        Home [H]
      </Button>
    </div>
  )
}
