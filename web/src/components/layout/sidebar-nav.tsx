import { Link, useRouterState } from '@tanstack/react-router'
import { Gamepad2, MessageSquare, Crosshair } from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/', label: 'Control', icon: Gamepad2 },
  { to: '/agent', label: 'Agent', icon: MessageSquare },
  { to: '/calibration', label: 'Calibrate', icon: Crosshair },
] as const

export function SidebarNav() {
  const routerState = useRouterState()
  const currentPath = routerState.location.pathname

  return (
    <nav className="flex w-14 flex-col items-center gap-2 border-r bg-muted/40 py-4">
      {navItems.map((item) => {
        const Icon = item.icon
        const active = currentPath === item.to
        return (
          <Link
            key={item.to}
            to={item.to}
            className={cn(
              'flex h-10 w-10 items-center justify-center rounded-lg transition-colors',
              active
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-muted hover:text-foreground',
            )}
            title={item.label}
          >
            <Icon className="h-5 w-5" />
          </Link>
        )
      })}
    </nav>
  )
}
