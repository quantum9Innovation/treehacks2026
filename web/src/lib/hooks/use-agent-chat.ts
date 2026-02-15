/** Hook for agent chat with WebSocket event streaming. */

import { useCallback, useState } from 'react'
import { agentApi } from '../api'
import type { ChatMessage, WSMessage } from '../types'

let msgIdCounter = 0
function nextId() {
  return `msg_${Date.now()}_${++msgIdCounter}`
}

export function useAgentChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isProcessing, setIsProcessing] = useState(false)

  const sendTask = useCallback(async (task: string) => {
    // Add user message
    const userMsg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content: task,
      timestamp: Date.now(),
    }
    setMessages((prev) => [...prev, userMsg])
    setIsProcessing(true)

    try {
      await agentApi.submitTask(task, true)
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: nextId(),
        role: 'agent',
        content: `Error: ${err instanceof Error ? err.message : 'Failed to submit task'}`,
        timestamp: Date.now(),
      }
      setMessages((prev) => [...prev, errorMsg])
      setIsProcessing(false)
    }
  }, [])

  const cancelTask = useCallback(async () => {
    try {
      await agentApi.cancel()
    } catch {
      // Ignore
    }
  }, [])

  const confirmAction = useCallback(async (approved: boolean) => {
    try {
      await agentApi.confirm(approved)
    } catch {
      // Ignore
    }
  }, [])

  const handleWSMessage = useCallback((msg: WSMessage) => {
    switch (msg.type) {
      case 'agent.iteration': {
        const text = msg.payload.message as string | undefined
        if (text) {
          setMessages((prev) => [
            ...prev,
            {
              id: nextId(),
              role: 'agent',
              content: text,
              timestamp: Date.now(),
            },
          ])
        }
        break
      }
      case 'agent.tool_call': {
        const status = msg.payload.status as string
        if (status === 'completed') {
          setMessages((prev) => [
            ...prev,
            {
              id: nextId(),
              role: 'tool_call',
              content: `${msg.payload.tool_name}(${JSON.stringify(msg.payload.args)})`,
              toolName: msg.payload.tool_name as string,
              toolArgs: msg.payload.args as Record<string, unknown>,
              toolResult: msg.payload.result as Record<string, unknown>,
              images: msg.payload.images as string[] | undefined,
              timestamp: Date.now(),
            },
          ])
        }
        break
      }
      case 'agent.task_complete': {
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: 'agent',
            content: msg.payload.result as string,
            timestamp: Date.now(),
          },
        ])
        setIsProcessing(false)
        break
      }
      case 'error': {
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: 'agent',
            content: `Error: ${msg.payload.message}`,
            timestamp: Date.now(),
          },
        ])
        setIsProcessing(false)
        break
      }
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    sendTask,
    cancelTask,
    confirmAction,
    isProcessing,
    handleWSMessage,
    clearMessages,
  }
}
