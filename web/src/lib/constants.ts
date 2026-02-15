/** Application constants. */

export const API_BASE = ''
export const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`

export const STEP_SIZES = [5, 10, 25, 50] as const
export const DEFAULT_STEP_SIZE = 25

export const CAMERA_WIDTH = 640
export const CAMERA_HEIGHT = 480
export const GRID_SPACING = 80
