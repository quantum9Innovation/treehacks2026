import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ChevronDown } from 'lucide-react'
import type { ChatMessage as ChatMessageType } from '@/lib/types'

interface ChatMessageProps {
  message: ChatMessageType
}

export function ChatMessage({ message }: ChatMessageProps) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.role === 'tool_call') {
    return (
      <Collapsible>
        <CollapsibleTrigger className="flex w-full items-center gap-2 rounded border px-2 py-1 text-xs hover:bg-muted">
          <Badge variant="outline" className="text-xs">
            {message.toolName}
          </Badge>
          <span className="flex-1 truncate text-left text-muted-foreground font-mono">
            {JSON.stringify(message.toolArgs)}
          </span>
          <ChevronDown className="h-3 w-3" />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-1 space-y-2 pl-4">
          {message.toolResult && (
            <pre className="overflow-auto rounded bg-muted p-2 text-xs font-mono">
              {JSON.stringify(message.toolResult, null, 2)}
            </pre>
          )}
          {message.images?.map((img, i) => (
            <img
              key={i}
              src={img}
              alt={`${message.toolName} result`}
              className="max-w-full rounded border"
            />
          ))}
        </CollapsibleContent>
      </Collapsible>
    )
  }

  // Agent message
  return (
    <div className="flex justify-start">
      <div className="max-w-[90%] rounded-lg bg-muted px-3 py-2 text-sm">
        {message.content}
      </div>
    </div>
  )
}
