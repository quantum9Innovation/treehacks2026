/** Keyboard shortcut hook for arm control (WASD/QE/RF/H/Space). */

import { useEffect } from 'react'
import { API_BASE } from '../constants'
import { useRobot } from '../robot-context'

async function postArm(path: string, body?: Record<string, unknown>) {
  try {
    await fetch(`${API_BASE}/api/arm/${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    })
  } catch (err) {
    console.error(`Arm command failed: ${path}`, err)
  }
}

export function useKeyboardControls(enabled: boolean) {
  const { state } = useRobot()

  useEffect(() => {
    if (!enabled) return

    const handler = (e: KeyboardEvent) => {
      // Skip if typing in an input
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      const pos = state.armPosition
      if (!pos) return

      const step = state.stepSize

      switch (e.key.toLowerCase()) {
        case 'w':
          postArm('move', { x: pos.x, y: pos.y, z: pos.z + step })
          break
        case 's':
          postArm('move', { x: pos.x, y: pos.y, z: Math.max(0, pos.z - step) })
          break
        case 'a':
          postArm('move', { x: pos.x, y: pos.y + step, z: pos.z })
          break
        case 'd':
          postArm('move', { x: pos.x, y: pos.y - step, z: pos.z })
          break
        case 'q':
          postArm('move', { x: pos.x + step, y: pos.y, z: pos.z })
          break
        case 'e':
          postArm('move', { x: pos.x - step, y: pos.y, z: pos.z })
          break
        case 'r':
          postArm('gripper', { angle: 90 })
          break
        case 'f':
          postArm('gripper', { angle: 0 })
          break
        case 'h':
          postArm('home')
          break
        case ' ':
          e.preventDefault()
          postArm('stop')
          break
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [enabled, state.armPosition, state.stepSize])
}
