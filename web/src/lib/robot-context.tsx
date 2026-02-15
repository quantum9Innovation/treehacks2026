/** Global robot state context with useReducer. */

import { createContext, useContext, useReducer, type ReactNode } from 'react'
import type { AgentState, CameraView, InteractionMode, Position } from './types'
import { DEFAULT_STEP_SIZE } from './constants'

interface RobotState {
  cameraConnected: boolean
  armConnected: boolean
  wsConnected: boolean
  armPosition: Position | null
  gripperAngle: number
  calibrationLoaded: boolean
  calibrationRMSE: number | null
  agentState: AgentState
  agentAction: string | null
  cameraView: CameraView
  interactionMode: InteractionMode
  stepSize: number
}

const initialState: RobotState = {
  cameraConnected: false,
  armConnected: false,
  wsConnected: false,
  armPosition: null,
  gripperAngle: 90,
  calibrationLoaded: false,
  calibrationRMSE: null,
  agentState: 'idle',
  agentAction: null,
  cameraView: 'color',
  interactionMode: 'off',
  stepSize: DEFAULT_STEP_SIZE,
}

type Action =
  | { type: 'SET_CAMERA_CONNECTED'; payload: boolean }
  | { type: 'SET_ARM_CONNECTED'; payload: boolean }
  | { type: 'SET_WS_CONNECTED'; payload: boolean }
  | { type: 'SET_ARM_POSITION'; payload: Position }
  | { type: 'SET_GRIPPER_ANGLE'; payload: number }
  | { type: 'SET_CALIBRATION'; payload: { loaded: boolean; rmse: number | null } }
  | { type: 'SET_AGENT_STATE'; payload: { state: AgentState; action?: string | null } }
  | { type: 'SET_CAMERA_VIEW'; payload: CameraView }
  | { type: 'SET_INTERACTION_MODE'; payload: InteractionMode }
  | { type: 'SET_STEP_SIZE'; payload: number }

function robotReducer(state: RobotState, action: Action): RobotState {
  switch (action.type) {
    case 'SET_CAMERA_CONNECTED':
      return { ...state, cameraConnected: action.payload }
    case 'SET_ARM_CONNECTED':
      return { ...state, armConnected: action.payload }
    case 'SET_WS_CONNECTED':
      return { ...state, wsConnected: action.payload }
    case 'SET_ARM_POSITION':
      return { ...state, armPosition: action.payload }
    case 'SET_GRIPPER_ANGLE':
      return { ...state, gripperAngle: action.payload }
    case 'SET_CALIBRATION':
      return { ...state, calibrationLoaded: action.payload.loaded, calibrationRMSE: action.payload.rmse }
    case 'SET_AGENT_STATE':
      return { ...state, agentState: action.payload.state, agentAction: action.payload.action ?? null }
    case 'SET_CAMERA_VIEW':
      return { ...state, cameraView: action.payload }
    case 'SET_INTERACTION_MODE':
      return { ...state, interactionMode: action.payload }
    case 'SET_STEP_SIZE':
      return { ...state, stepSize: action.payload }
    default:
      return state
  }
}

const RobotContext = createContext<{
  state: RobotState
  dispatch: React.Dispatch<Action>
} | null>(null)

export function RobotProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(robotReducer, initialState)
  return (
    <RobotContext.Provider value={{ state, dispatch }}>
      {children}
    </RobotContext.Provider>
  )
}

export function useRobot() {
  const ctx = useContext(RobotContext)
  if (!ctx) throw new Error('useRobot must be used within RobotProvider')
  return ctx
}
