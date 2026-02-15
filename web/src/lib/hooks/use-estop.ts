/** Global e-stop hook: Space key fires emergency stop on ALL pages. */

import { useEffect } from 'react'
import { API_BASE } from '../constants'

function fireEstop() {
  fetch(`${API_BASE}/api/arm/stop`, { method: 'POST' }).catch((err) =>
    console.error('E-STOP failed:', err),
  )
}

export { fireEstop }

export function useEstop() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== ' ') return
      // Always prevent default so Space never opens dropdowns / scrolls
      e.preventDefault()
      e.stopPropagation()
      fireEstop()
    }

    // Use capture phase so we beat any component-level listeners
    window.addEventListener('keydown', handler, { capture: true })
    return () => window.removeEventListener('keydown', handler, { capture: true })
  }, [])
}
