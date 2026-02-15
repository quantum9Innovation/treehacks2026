import { Button } from '@/components/ui/button'
import { armApi } from '@/lib/api'
import { OctagonX } from 'lucide-react'

export function EmergencyStop() {
  return (
    <Button
      variant="destructive"
      size="lg"
      className="w-full text-sm font-bold"
      onClick={() => armApi.stop()}
    >
      <OctagonX className="mr-1 h-4 w-4" />
      E-STOP [Space]
    </Button>
  )
}
