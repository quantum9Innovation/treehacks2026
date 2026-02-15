import { useRef, useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { useRobot } from '@/lib/robot-context'
import { ChatMessage } from './chat-message'
import { Send, Square, Check, X } from 'lucide-react'
import type { ChatMessage as ChatMessageType } from '@/lib/types'

interface ChatContainerProps {
  messages: ChatMessageType[]
  isProcessing: boolean
  onSendTask: (task: string) => void
  onCancel: () => void
  onConfirm: (approved: boolean) => void
}

export function ChatContainer({
  messages,
  isProcessing,
  onSendTask,
  onCancel,
  onConfirm,
}: ChatContainerProps) {
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const { state } = useRobot()

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isProcessing) return
    onSendTask(trimmed)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <ScrollArea className="flex-1 px-3 py-2" ref={scrollRef}>
        <div className="space-y-3">
          {messages.length === 0 && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              Enter a task for the agent, e.g. "pick up the red cup"
            </p>
          )}
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          {isProcessing && state.agentState !== 'awaiting_confirm' && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="animate-pulse">Thinking...</span>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Confirmation buttons */}
      {state.agentState === 'awaiting_confirm' && (
        <div className="flex gap-2 border-t px-3 py-2">
          <span className="flex-1 text-xs text-muted-foreground self-center">
            {state.agentAction || 'Approve action?'}
          </span>
          <Button size="sm" onClick={() => onConfirm(true)}>
            <Check className="mr-1 h-3 w-3" />
            Approve
          </Button>
          <Button size="sm" variant="destructive" onClick={() => onConfirm(false)}>
            <X className="mr-1 h-3 w-3" />
            Reject
          </Button>
        </div>
      )}

      {/* Input */}
      <div className="flex gap-2 border-t p-3">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe a task..."
          className="min-h-[40px] max-h-[80px] resize-none text-sm"
          disabled={isProcessing}
        />
        {isProcessing ? (
          <Button variant="destructive" size="icon" onClick={onCancel}>
            <Square className="h-4 w-4" />
          </Button>
        ) : (
          <Button size="icon" onClick={handleSend} disabled={!input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  )
}
