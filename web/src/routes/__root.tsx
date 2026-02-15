import { Outlet, createRootRoute } from '@tanstack/react-router'

import { StatusBar } from '@/components/layout/status-bar'
import { SidebarNav } from '@/components/layout/sidebar-nav'
import { RobotProvider } from '@/lib/robot-context'
import { TooltipProvider } from '@/components/ui/tooltip'

import '../styles.css'

export const Route = createRootRoute({
  component: RootLayout,
})

function RootLayout() {
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
        </div>
      </TooltipProvider>
    </RobotProvider>
  )
}
