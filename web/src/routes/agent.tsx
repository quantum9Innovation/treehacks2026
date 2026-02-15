import { createFileRoute } from '@tanstack/react-router'
import { CameraFeed } from '@/components/camera/camera-feed'
import { ChatContainer } from '@/components/chat/chat-container'
import { useWebSocket } from '@/lib/hooks/use-websocket'
import { useAgentChat } from '@/lib/hooks/use-agent-chat'

export const Route = createFileRoute('/agent')({
  component: AgentPage,
})

function AgentPage() {
  const {
    messages,
    sendTask,
    cancelTask,
    confirmAction,
    isProcessing,
    handleWSMessage,
  } = useAgentChat()

  useWebSocket(handleWSMessage)

  return (
    <div className="flex h-full gap-4 p-4">
      {/* Camera panel */}
      <div className="flex-[3] min-w-0">
        <CameraFeed showToolbar={false} />
      </div>

      {/* Chat panel */}
      <div className="flex w-96 flex-col border-l">
        <ChatContainer
          messages={messages}
          isProcessing={isProcessing}
          onSendTask={sendTask}
          onCancel={cancelTask}
          onConfirm={confirmAction}
        />
      </div>
    </div>
  )
}
