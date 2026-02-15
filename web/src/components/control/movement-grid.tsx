import { Button } from '@/components/ui/button'
import { useRobot } from '@/lib/robot-context'
import { armApi } from '@/lib/api'
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, MoveUp, MoveDown } from 'lucide-react'

export function MovementGrid() {
  const { state } = useRobot()
  const pos = state.armPosition

  const move = (dx: number, dy: number, dz: number) => {
    if (!pos) return
    armApi.move(pos.x + dx, pos.y + dy, Math.max(0, pos.z + dz))
  }

  const step = state.stepSize

  return (
    <div className="grid grid-cols-3 gap-1">
      {/* Row 1 */}
      <Button variant="outline" size="sm" onClick={() => move(step, 0, 0)} title="X+ (Forward) [Q]">
        <MoveUp className="h-4 w-4" />
      </Button>
      <Button variant="outline" size="sm" onClick={() => move(0, 0, step)} title="Z+ (Up) [W]">
        <ArrowUp className="h-4 w-4" />
      </Button>
      <Button variant="outline" size="sm" onClick={() => move(-step, 0, 0)} title="X- (Back) [E]">
        <MoveDown className="h-4 w-4" />
      </Button>

      {/* Row 2 */}
      <Button variant="outline" size="sm" onClick={() => move(0, step, 0)} title="Y+ (Left) [A]">
        <ArrowLeft className="h-4 w-4" />
      </Button>
      <Button variant="outline" size="sm" onClick={() => move(0, 0, -step)} title="Z- (Down) [S]">
        <ArrowDown className="h-4 w-4" />
      </Button>
      <Button variant="outline" size="sm" onClick={() => move(0, -step, 0)} title="Y- (Right) [D]">
        <ArrowRight className="h-4 w-4" />
      </Button>
    </div>
  )
}
