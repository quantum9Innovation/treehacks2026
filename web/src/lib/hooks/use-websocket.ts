/** WebSocket hook for real-time event bus connection. */

import { useCallback, useEffect, useRef } from 'react'
import { WS_URL } from '../constants'
import type { WSMessage } from '../types'
import { useRobot } from '../robot-context'

export function useWebSocket(onMessage?: (msg: WSMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null)
  const { dispatch } = useRobot()
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>()

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      dispatch({ type: 'SET_WS_CONNECTED', payload: true })
    }

    ws.onclose = () => {
      dispatch({ type: 'SET_WS_CONNECTED', payload: false })
      // Reconnect after 2s
      reconnectTimeout.current = setTimeout(connect, 2000)
    }

    ws.onerror = () => {
      ws.close()
    }

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data)

        // Dispatch known events to global state
        switch (msg.type) {
          case 'arm.position':
            dispatch({
              type: 'SET_ARM_POSITION',
              payload: msg.payload as { x: number; y: number; z: number },
            })
            break
          case 'arm.gripper':
            dispatch({
              type: 'SET_GRIPPER_ANGLE',
              payload: msg.payload.angle as number,
            })
            break
          case 'arm.connected':
            dispatch({
              type: 'SET_ARM_CONNECTED',
              payload: { connected: true, device: msg.payload.device as string },
            })
            break
          case 'arm.disconnected':
            dispatch({
              type: 'SET_ARM_CONNECTED',
              payload: { connected: false, device: null },
            })
            break
          case 'arm.ground_calibrated':
            if (msg.payload.device) {
              dispatch({
                type: 'SET_GROUND_CALIBRATED',
                payload: msg.payload.device as string,
              })
            }
            break
          case 'agent.state_changed':
            dispatch({
              type: 'SET_AGENT_STATE',
              payload: {
                state: msg.payload.state as 'idle' | 'running' | 'awaiting_confirm' | 'error',
                action: (msg.payload.action as string) ?? null,
              },
            })
            break
        }

        // Forward to custom handler
        onMessage?.(msg)
      } catch {
        // Ignore parse errors
      }
    }
  }, [dispatch, onMessage])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimeout.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { send }
}
