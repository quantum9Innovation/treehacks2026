import { HeadContent, Outlet, Scripts, createRootRoute } from '@tanstack/react-router'

import { StatusBar } from '@/components/layout/status-bar'
import { SidebarNav } from '@/components/layout/sidebar-nav'
import { RobotProvider } from '@/lib/robot-context'
import { TooltipProvider } from '@/components/ui/tooltip'

import appCss from '../styles.css?url'

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'RoArm Control' },
    ],
    links: [{ rel: 'stylesheet', href: appCss }],
  }),
  shellComponent: RootDocument,
})

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <HeadContent />
      </head>
      <body className="min-h-screen bg-background text-foreground antialiased">
        <RobotProvider>
          <TooltipProvider>
            <div className="flex h-screen flex-col">
              <StatusBar />
              <div className="flex flex-1 overflow-hidden">
                <SidebarNav />
                <main className="flex-1 overflow-auto">
                  {children}
                  <Outlet />
                </main>
              </div>
            </div>
          </TooltipProvider>
        </RobotProvider>
        <Scripts />
      </body>
    </html>
  )
}
