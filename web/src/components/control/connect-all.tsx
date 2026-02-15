import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { armApi } from '@/lib/api'
import { useRobot } from '@/lib/robot-context'
import { Zap } from 'lucide-react'

export function ConnectAll() {
  const [loading, setLoading] = useState(false)
  const { state } = useRobot()

  const allProbed =
    state.groundProbedDevices.length >= 4 && state.armConnected

  const handleConnectAll = async () => {
    setLoading(true)
    try {
      const res = await armApi.connectAndProbeAll()
      if (res.failed.length > 0) {
        console.warn('Some arms failed:', res.failed)
      }
    } catch (e) {
      console.error('Connect and probe all failed:', e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Button
      variant={allProbed ? 'outline' : 'default'}
      size="lg"
      className="w-full text-sm"
      onClick={handleConnectAll}
      disabled={loading}
    >
      <Zap className="mr-1 h-4 w-4" />
      {loading ? 'Connecting & Probing...' : allProbed ? 'Reconnect All' : 'Connect & Probe All'}
    </Button>
  )
}
