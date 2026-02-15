import { Button } from '@/components/ui/button'
import { fireEstop } from '@/lib/hooks/use-estop'
import { OctagonX } from 'lucide-react'

export function EmergencyStop({ className }: { className?: string }) {
  return (
    <Button
      variant="destructive"
      size="lg"
      className={className ?? 'w-full text-sm font-bold'}
      onClick={() => fireEstop()}
    >
      <OctagonX className="mr-1 h-4 w-4" />
      E-STOP [Space]
    </Button>
  )
}
