/** Shared TypeScript interfaces for the robot arm control frontend. */

export interface Position {
  x: number
  y: number
  z: number
}

export interface WSMessage {
  type: string
  payload: Record<string, unknown>
  timestamp: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'agent' | 'tool_call' | 'tool_result'
  content: string
  images?: string[]
  toolName?: string
  toolArgs?: Record<string, unknown>
  toolResult?: Record<string, unknown>
  timestamp: number
}

export interface SegmentResult {
  score: number
  bbox: { x1: number; y1: number; x2: number; y2: number }
  mask_area_px: number
  depth_mm: number | null
  arm_coordinates: Position | null
  annotated_b64: string
}

export type AgentState = 'idle' | 'running' | 'awaiting_confirm' | 'error'
export type InteractionMode = 'off' | 'segment' | 'touch'
export type CameraView = 'color' | 'depth'
