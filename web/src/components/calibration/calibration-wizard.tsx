import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { calibrationApi } from '@/lib/api'

interface CalibrationState {
  active: boolean
  step: number
  totalSteps: number
  pointCount: number
  position: { x: number; y: number; z: number } | null
  status: string
  result: {
    rmse_mm: number
    quality: string
    points_used: number
  } | null
}

interface CalibrationWizardProps {
  onClickPixel: (handler: ((px: number, py: number) => void) | null) => void
}

export function CalibrationWizard({ onClickPixel }: CalibrationWizardProps) {
  const [cal, setCal] = useState<CalibrationState>({
    active: false,
    step: 0,
    totalSteps: 0,
    pointCount: 0,
    position: null,
    status: 'idle',
    result: null,
  })

  const startCalibration = async () => {
    try {
      const res = await calibrationApi.start()
      setCal({
        active: true,
        step: 1,
        totalSteps: res.total_steps,
        pointCount: 0,
        position: null,
        status: 'waiting_for_click',
        result: null,
      })
      // Register click handler
      onClickPixel(handlePixelClick)
    } catch (err) {
      console.error('Failed to start calibration:', err)
    }
  }

  const handlePixelClick = async (pixel_x: number, pixel_y: number) => {
    try {
      const res = await calibrationApi.click(pixel_x, pixel_y)
      if (res.status === 'done') {
        setCal((prev) => ({
          ...prev,
          active: false,
          result: { rmse_mm: res.rmse_mm!, quality: res.quality!, points_used: res.points_used! },
          status: 'done',
        }))
        onClickPixel(null)
      } else {
        setCal((prev) => ({
          ...prev,
          step: prev.step + 1,
          pointCount: res.point_count ?? prev.pointCount,
        }))
      }
    } catch (err) {
      console.error('Calibration click failed:', err)
    }
  }

  const handleSkip = async () => {
    try {
      const res = await calibrationApi.skip()
      if (res.status === 'done') {
        setCal((prev) => ({
          ...prev,
          active: false,
          result: { rmse_mm: res.rmse_mm!, quality: res.quality!, points_used: res.points_used! },
          status: 'done',
        }))
        onClickPixel(null)
      } else {
        setCal((prev) => ({
          ...prev,
          step: prev.step + 1,
          pointCount: res.point_count ?? prev.pointCount,
        }))
      }
    } catch {
      /* ignore */
    }
  }

  const handleAbort = async () => {
    try {
      await calibrationApi.abort()
    } catch {
      /* ignore */
    }
    setCal((prev) => ({ ...prev, active: false, status: 'idle' }))
    onClickPixel(null)
  }

  if (!cal.active && !cal.result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Camera-to-Arm Calibration</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-3 text-xs text-muted-foreground">
            8-point calibration: the arm moves to known positions and you click on the arm tip in
            the camera feed.
          </p>
          <Button onClick={startCalibration} size="sm">
            Start Calibration
          </Button>
        </CardContent>
      </Card>
    )
  }

  if (cal.result) {
    const qualityColor =
      cal.result.quality === 'excellent'
        ? 'text-green-600'
        : cal.result.quality === 'good'
          ? 'text-yellow-600'
          : 'text-red-600'

    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Calibration Complete</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="text-center">
            <div className={`text-3xl font-bold ${qualityColor}`}>
              {cal.result.rmse_mm.toFixed(1)}mm
            </div>
            <Badge
              variant={cal.result.quality === 'excellent' ? 'default' : 'secondary'}
              className="mt-1"
            >
              {cal.result.quality}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground text-center">
            {cal.result.points_used} points used
          </p>
          <Button variant="outline" size="sm" className="w-full" onClick={() => setCal((prev) => ({ ...prev, result: null }))}>
            New Calibration
          </Button>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">
          Calibrating — Point {cal.step}/{cal.totalSteps}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Progress value={(cal.step / cal.totalSteps) * 100} />
        {cal.position && (
          <p className="text-xs text-muted-foreground font-mono">
            Arm at: ({cal.position.x.toFixed(0)}, {cal.position.y.toFixed(0)},{' '}
            {cal.position.z.toFixed(0)})
          </p>
        )}
        <p className="text-xs">Click on the arm tip in the camera feed.</p>
        <p className="text-xs text-muted-foreground">{cal.pointCount} points collected</p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleSkip}>
            Skip
          </Button>
          <Button variant="destructive" size="sm" onClick={handleAbort}>
            Abort
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
