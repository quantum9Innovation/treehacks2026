/** HTTP API helpers for the Python backend. */

import { API_BASE } from './constants'

async function post<T = unknown>(path: string, body?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

async function get<T = unknown>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

// --- Arm ---

export const armApi = {
  devices: () =>
    get<{ devices: string[]; active: string | null }>('/api/arm/devices'),
  connect: (device: string) =>
    post<{ status: string; device: string; message: string }>('/api/arm/connect', { device }),
  disconnect: () =>
    post<{ status: string; device: string; message: string }>('/api/arm/disconnect'),
  pose: () => get<{ x: number; y: number; z: number }>('/api/arm/pose'),
  move: (x: number, y: number, z: number) => post('/api/arm/move', { x, y, z }),
  home: () => post('/api/arm/home'),
  stop: () => post('/api/arm/stop'),
  probeGround: () => post<{ status: string; message: string }>('/api/arm/probe-ground'),
  gripper: (angle: number) => post('/api/arm/gripper', { angle }),
  gotoPixel: (pixel_x: number, pixel_y: number, z_offset_mm = 50) =>
    post('/api/arm/goto-pixel', { pixel_x, pixel_y, z_offset_mm }),
  status: () =>
    get<{ connected: boolean; ground_calibrated: boolean; active_device: string | null }>('/api/arm/status'),
}

// --- Vision ---

export const visionApi = {
  look: () =>
    post<{
      color_b64: string
      depth_b64: string
      width: number
      height: number
      sam2_encode_time_s: number
    }>('/api/vision/look'),
  segment: (pixel_x: number, pixel_y: number) =>
    post<{
      score: number
      bbox: { x1: number; y1: number; x2: number; y2: number }
      mask_area_px: number
      depth_mm: number | null
      arm_coordinates: { x: number; y: number; z: number } | null
      annotated_b64: string
    }>('/api/vision/segment', { pixel_x, pixel_y }),
}

// --- Agent ---

export const agentApi = {
  submitTask: (task: string, auto_confirm = false) =>
    post<{ task_id: string; status: string }>('/api/agent/task', { task, auto_confirm }),
  cancel: () => post('/api/agent/cancel'),
  confirm: (approved: boolean) => post('/api/agent/confirm', { approved }),
  status: () =>
    get<{ state: string; task: string | null }>('/api/agent/status'),
}

// --- Calibration ---

export const calibrationApi = {
  status: () =>
    get<{
      has_calibration: boolean
      rmse_mm: number | null
      session_active: boolean
      current_step: number
      total_steps: number
    }>('/api/calibration/status'),
  start: (mode = 'manual') =>
    post<{ status: string; total_steps: number }>('/api/calibration/start', { mode }),
  click: (pixel_x: number, pixel_y: number) =>
    post<{ status: string; point_count?: number; rmse_mm?: number; quality?: string; points_used?: number }>(
      '/api/calibration/click',
      { pixel_x, pixel_y },
    ),
  skip: () =>
    post<{ status: string; point_count?: number; rmse_mm?: number; quality?: string; points_used?: number }>(
      '/api/calibration/skip',
    ),
  abort: () => post<{ status: string; points_collected: number }>('/api/calibration/abort'),
}

// --- Camera ---

export const cameraApi = {
  snapshot: () =>
    get<{ color_b64: string; depth_b64: string; width: number; height: number }>('/api/camera/snapshot'),
}

// --- Health ---

export const healthApi = {
  check: () => get<{ status: string }>('/api/health'),
}
