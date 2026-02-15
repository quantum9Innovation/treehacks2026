import { Outlet, createRootRoute } from '@tanstack/react-router'

import { StatusBar } from '@/components/layout/status-bar'
import { SidebarNav } from '@/components/layout/sidebar-nav'
import { EmergencyStop } from '@/components/control/emergency-stop'
import { RobotProvider } from '@/lib/robot-context'
import { TooltipProvider } from '@/components/ui/tooltip'
import { useEstop } from '@/lib/hooks/use-estop'

import '../styles.css'

export const Route = createRootRoute({
  component: RootLayout,
})

function RootLayout() {
  // Global Space-key e-stop listener (capture phase, all pages)
  useEstop()

  return (
    <RobotProvider>
      <TooltipProvider>
        <div className="flex h-screen flex-col">
          <StatusBar />
          <div className="flex flex-1 overflow-hidden">
            <SidebarNav />
            <main className="flex-1 overflow-auto">
              <Outlet />
            </main>
          </div>
          {/* Floating e-stop visible on every page */}
          <div className="fixed bottom-4 right-4 z-50">
            <EmergencyStop className="text-sm font-bold shadow-lg" />
          </div>
        </div>
      </TooltipProvider>
    </RobotProvider>
  )
}
