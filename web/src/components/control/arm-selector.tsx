import { useEffect, useState } from 'react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { armApi } from '@/lib/api'
import { useRobot } from '@/lib/robot-context'
import { Unplug } from 'lucide-react'

export function ArmSelector() {
  const { state, dispatch } = useRobot()
  const [devices, setDevices] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  // Fetch available devices + current active on mount
  useEffect(() => {
    armApi.devices().then((res) => {
      setDevices(res.devices)
      if (res.active) {
        dispatch({
          type: 'SET_ARM_CONNECTED',
          payload: { connected: true, device: res.active },
        })
      }
    }).catch(() => {})
  }, [dispatch])

  const handleSelect = async (device: string) => {
    if (device === state.activeDevice) return
    setLoading(true)
    try {
      await armApi.connect(device)
    } catch (e) {
      console.error('Arm connect failed:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleDisconnect = async () => {
    setLoading(true)
    try {
      await armApi.disconnect()
    } catch (e) {
      console.error('Arm disconnect failed:', e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <Select
        value={state.activeDevice ?? ''}
        onValueChange={handleSelect}
        disabled={loading}
      >
        <SelectTrigger size="sm" className="flex-1">
          <SelectValue placeholder={loading ? 'Connecting...' : 'Select arm'} />
        </SelectTrigger>
        <SelectContent>
          {devices.map((d) => (
            <SelectItem key={d} value={d}>
              {d.replace('/dev/', '')}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {state.armConnected && (
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={handleDisconnect}
          disabled={loading}
          title="Disconnect arm"
        >
          <Unplug className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
