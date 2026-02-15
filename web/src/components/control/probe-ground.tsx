import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { armApi } from '@/lib/api'
import { useGroundCalibrated } from '@/lib/robot-context'
import { Crosshair } from 'lucide-react'

export function ProbeGround() {
  const [loading, setLoading] = useState(false)
  const groundCalibrated = useGroundCalibrated()

  const handleProbe = async () => {
    setLoading(true)
    try {
      await armApi.probeGround()
    } catch (e) {
      console.error('Probe ground failed:', e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button
      variant={groundCalibrated ? 'outline' : 'default'}
      size="lg"
      className="w-full text-sm"
      onClick={handleProbe}
      disabled={loading}
    >
      <Crosshair className="mr-1 h-4 w-4" />
      {loading ? 'Probing...' : groundCalibrated ? 'Re-probe Ground' : 'Probe Ground'}
    </Button>
  )
}
