import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { calibrationApi } from '@/lib/api'
import type { WSMessage } from '@/lib/types'

type CalibrationMode = 'manual' | 'aruco' | 'aruco_all'

interface ArmResult {
  arm_name: string
  rmse_mm: number
  quality: string
  points_used: number
  status: string
}

interface CalibrationState {
  active: boolean
  mode: CalibrationMode
  step: number
  totalSteps: number
  pointCount: number
  position: { x: number; y: number; z: number } | null
  status: string
  settleCount: number
  settleTarget: number
  // Multi-arm tracking
  armIndex: number
  armTotal: number
  device: string | null
  armResults: ArmResult[]
  // Final result (single-arm or summary)
  result: {
    rmse_mm: number
    quality: string
    points_used: number
  } | null
  // Multi-arm final results
  multiArmResult: Record<string, ArmResult> | null
}

interface CalibrationWizardProps {
  onClickPixel: (handler: ((px: number, py: number) => void) | null) => void
  lastWsMessage?: WSMessage | null
}

export function CalibrationWizard({ onClickPixel, lastWsMessage }: CalibrationWizardProps) {
  const [mode, setMode] = useState<CalibrationMode>('aruco')
  const [cal, setCal] = useState<CalibrationState>({
    active: false,
    mode: 'manual',
    step: 0,
    totalSteps: 0,
    pointCount: 0,
    position: null,
    status: 'idle',
    settleCount: 0,
    settleTarget: 5,
    armIndex: 0,
    armTotal: 0,
    device: null,
    armResults: [],
    result: null,
    multiArmResult: null,
  })
  const onClickPixelRef = useRef(onClickPixel)
  onClickPixelRef.current = onClickPixel

  // Handle calibration WS events
  useEffect(() => {
    if (!lastWsMessage) return
    const msg = lastWsMessage

    if (msg.type === 'calibration.progress') {
      const p = msg.payload
      setCal((prev) => ({
        ...prev,
        step: (p.step as number) ?? prev.step,
        totalSteps: (p.total_steps as number) ?? prev.totalSteps,
        pointCount: (p.point_count as number) ?? prev.pointCount,
        status: (p.status as string) ?? prev.status,
        settleCount: (p.settle_count as number) ?? prev.settleCount,
        settleTarget: (p.settle_target as number) ?? prev.settleTarget,
        position: p.position
          ? (p.position as { x: number; y: number; z: number })
          : prev.position,
        // Multi-arm fields
        armIndex: (p.arm_index as number) ?? prev.armIndex,
        armTotal: (p.arm_total as number) ?? prev.armTotal,
        device: (p.device as string) ?? prev.device,
      }))
    } else if (msg.type === 'calibration.arm_result') {
      // Per-arm result during multi-arm calibration
      const p = msg.payload
      const armResult: ArmResult = {
        arm_name: p.arm_name as string,
        rmse_mm: p.rmse_mm as number,
        quality: p.quality as string,
        points_used: p.points_used as number,
        status: p.status as string,
      }
      setCal((prev) => ({
        ...prev,
        armResults: [...prev.armResults, armResult],
      }))
    } else if (msg.type === 'calibration.result') {
      const p = msg.payload
      if (p.status === 'done') {
        // Check if this is a multi-arm result (has 'arms' field)
        if (p.arms) {
          const arms = p.arms as Record<string, { rmse_mm: number; quality: string; points_used: number; status: string }>
          setCal((prev) => ({
            ...prev,
            active: false,
            status: 'done',
            multiArmResult: Object.fromEntries(
              Object.entries(arms).map(([name, r]) => [
                name,
                { arm_name: name, ...r },
              ]),
            ),
          }))
        } else {
          setCal((prev) => ({
            ...prev,
            active: false,
            status: 'done',
            result: {
              rmse_mm: p.rmse_mm as number,
              quality: p.quality as string,
              points_used: p.points_used as number,
            },
          }))
        }
        onClickPixelRef.current(null)
      } else if (p.status === 'aborted' || p.status === 'error') {
        setCal((prev) => ({
          ...prev,
          active: false,
          status: 'idle',
        }))
        onClickPixelRef.current(null)
      }
    }
  }, [lastWsMessage])

  const startCalibration = async () => {
    try {
      const res = await calibrationApi.start(mode)
      setCal({
        active: true,
        mode,
        step: 1,
        totalSteps: res.total_steps,
        pointCount: 0,
        position: null,
        status: mode === 'manual' ? 'waiting_for_click' : 'moving',
        settleCount: 0,
        settleTarget: 5,
        armIndex: 0,
        armTotal: (res as Record<string, unknown>).arms as number ?? 0,
        device: null,
        armResults: [],
        result: null,
        multiArmResult: null,
      })
      if (mode === 'manual') {
        onClickPixel(handlePixelClick)
      }
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
      } else if (res.status !== 'skip_requested') {
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

  const resetState = () =>
    setCal((prev) => ({ ...prev, result: null, multiArmResult: null, armResults: [] }))

  // Idle state — show mode selector + start button
  if (!cal.active && !cal.result && !cal.multiArmResult) {
    const modeDescriptions: Record<CalibrationMode, string> = {
      manual:
        '8-point calibration: the arm moves to known positions and you click on the arm tip in the camera feed.',
      aruco:
        '8-point calibration: the arm moves to known positions and ArUco markers are detected automatically.',
      aruco_all:
        'Calibrate ARM1, ARM2, ARM3 sequentially using fixed ArUco marker assignments (ARM1=0, ARM2=1, ARM3=2).',
    }

    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Camera-to-Arm Calibration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">{modeDescriptions[mode]}</p>
          <div className="flex gap-1 rounded-md border p-0.5">
            <button
              className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
                mode === 'aruco' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
              }`}
              onClick={() => setMode('aruco')}
            >
              ArUco
            </button>
            <button
              className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
                mode === 'aruco_all' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
              }`}
              onClick={() => setMode('aruco_all')}
            >
              All Arms
            </button>
            <button
              className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors ${
                mode === 'manual' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
              }`}
              onClick={() => setMode('manual')}
            >
              Manual
            </button>
          </div>
          <Button onClick={startCalibration} size="sm" className="w-full">
            Start Calibration
          </Button>
        </CardContent>
      </Card>
    )
  }

  // Multi-arm result state
  if (cal.multiArmResult) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Multi-Arm Calibration Complete</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {Object.entries(cal.multiArmResult).map(([name, r]) => {
            const qualityColor =
              r.quality === 'excellent'
                ? 'text-green-600'
                : r.quality === 'good'
                  ? 'text-yellow-600'
                  : 'text-red-600'
            return (
              <div key={name} className="flex items-center justify-between rounded border p-2">
                <span className="text-xs font-medium">{name}</span>
                {r.status === 'done' ? (
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-bold ${qualityColor}`}>
                      {r.rmse_mm.toFixed(1)}mm
                    </span>
                    <Badge
                      variant={r.quality === 'excellent' ? 'default' : 'secondary'}
                      className="text-[10px]"
                    >
                      {r.quality}
                    </Badge>
                    <span className="text-[10px] text-muted-foreground">
                      {r.points_used}pts
                    </span>
                  </div>
                ) : (
                  <Badge variant="destructive" className="text-[10px]">
                    failed
                  </Badge>
                )}
              </div>
            )
          })}
          <Button variant="outline" size="sm" className="w-full" onClick={resetState}>
            New Calibration
          </Button>
        </CardContent>
      </Card>
    )
  }

  // Single-arm result state
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
          <Button variant="outline" size="sm" className="w-full" onClick={resetState}>
            New Calibration
          </Button>
        </CardContent>
      </Card>
    )
  }

  // Active calibration state
  const isAuto = cal.mode === 'aruco' || cal.mode === 'aruco_all'
  const isMultiArm = cal.mode === 'aruco_all'

  const modeLabel =
    cal.mode === 'aruco_all' ? 'All Arms' : cal.mode === 'aruco' ? 'ArUco' : 'Manual'

  // Multi-arm: show which arm + device
  const armLabel = isMultiArm && cal.device ? ` ${cal.device.split('/').pop()}` : ''

  const statusText =
    cal.status === 'moving'
      ? 'Moving arm...'
      : cal.status === 'detecting'
        ? `Detecting marker... (${cal.settleCount}/${cal.settleTarget})`
        : cal.status === 'timeout'
          ? 'No marker found, skipping...'
          : cal.status === 'waiting_for_click'
            ? 'Click on the arm tip in the camera feed.'
            : cal.status

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">
          Calibrating ({modeLabel}{armLabel}) — Point {cal.step}/{cal.totalSteps}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Progress value={(cal.step / cal.totalSteps) * 100} />
        {isMultiArm && cal.armTotal > 0 && (
          <p className="text-xs font-medium">
            Arm {cal.armIndex + 1}/{cal.armTotal}
            {cal.device && ` — ${cal.device.split('/').pop()}`}
          </p>
        )}
        {cal.position && (
          <p className="text-xs text-muted-foreground font-mono">
            Arm at: ({cal.position.x.toFixed(0)}, {cal.position.y.toFixed(0)},{' '}
            {cal.position.z.toFixed(0)})
          </p>
        )}
        <p className="text-xs">{statusText}</p>
        {isAuto && cal.status === 'detecting' && (
          <Progress value={(cal.settleCount / cal.settleTarget) * 100} className="h-1" />
        )}
        <p className="text-xs text-muted-foreground">{cal.pointCount} points collected</p>
        {/* Show per-arm results as they complete */}
        {isMultiArm && cal.armResults.length > 0 && (
          <div className="space-y-1">
            {cal.armResults.map((r) => (
              <div
                key={r.arm_name}
                className="flex items-center justify-between text-[10px] rounded border px-2 py-1"
              >
                <span className="font-medium">{r.arm_name}</span>
                {r.status === 'done' ? (
                  <span
                    className={
                      r.quality === 'excellent'
                        ? 'text-green-600'
                        : r.quality === 'good'
                          ? 'text-yellow-600'
                          : 'text-red-600'
                    }
                  >
                    {r.rmse_mm.toFixed(1)}mm ({r.quality})
                  </span>
                ) : (
                  <span className="text-red-600">failed</span>
                )}
              </div>
            ))}
          </div>
        )}
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
