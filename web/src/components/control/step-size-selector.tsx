import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useRobot } from '@/lib/robot-context'
import { STEP_SIZES } from '@/lib/constants'

export function StepSizeSelector() {
  const { state, dispatch } = useRobot()

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">Step</span>
      <Select
        value={state.stepSize.toString()}
        onValueChange={(v) => dispatch({ type: 'SET_STEP_SIZE', payload: Number(v) })}
      >
        <SelectTrigger className="h-7 w-20 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {STEP_SIZES.map((s) => (
            <SelectItem key={s} value={s.toString()} className="text-xs">
              {s}mm
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
