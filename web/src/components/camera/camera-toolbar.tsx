import { Button } from '@/components/ui/button'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useRobot } from '@/lib/robot-context'
import { visionApi } from '@/lib/api'
import { Grid3X3, Eye, Hand, X, Camera } from 'lucide-react'
import type { InteractionMode, CameraView } from '@/lib/types'

interface CameraToolbarProps {
  showGrid: boolean
  onToggleGrid: () => void
  onClearOverlay: () => void
}

export function CameraToolbar({ showGrid, onToggleGrid, onClearOverlay }: CameraToolbarProps) {
  const { state, dispatch } = useRobot()

  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-2">
        {/* View toggle */}
        <ToggleGroup
          type="single"
          value={state.cameraView}
          onValueChange={(v) => {
            if (v) dispatch({ type: 'SET_CAMERA_VIEW', payload: v as CameraView })
          }}
          size="sm"
        >
          <ToggleGroupItem value="color" className="text-xs">
            Color
          </ToggleGroupItem>
          <ToggleGroupItem value="depth" className="text-xs">
            Depth
          </ToggleGroupItem>
        </ToggleGroup>

        {/* Mode selector */}
        <ToggleGroup
          type="single"
          value={state.interactionMode}
          onValueChange={(v) => {
            if (v) dispatch({ type: 'SET_INTERACTION_MODE', payload: v as InteractionMode })
          }}
          size="sm"
        >
          <ToggleGroupItem value="off" className="text-xs">
            Off
          </ToggleGroupItem>
          <ToggleGroupItem value="segment" className="text-xs">
            <Eye className="mr-1 h-3 w-3" />
            Segment
          </ToggleGroupItem>
          <ToggleGroupItem value="touch" className="text-xs">
            <Hand className="mr-1 h-3 w-3" />
            Touch
          </ToggleGroupItem>
        </ToggleGroup>
      </div>

      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" onClick={onToggleGrid} title="Toggle grid">
          <Grid3X3 className={`h-4 w-4 ${showGrid ? 'text-foreground' : 'text-muted-foreground'}`} />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={async () => {
            try {
              await visionApi.look()
            } catch {
              /* ignore */
            }
          }}
          title="Capture frame for SAM2"
        >
          <Camera className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onClearOverlay} title="Clear overlay">
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
