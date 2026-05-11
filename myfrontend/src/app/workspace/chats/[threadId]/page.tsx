"use client";

import { useParams } from "next/navigation";

import { MessageList } from "@/components/workspace/messages/message-list";
import { InputBox } from "@/components/workspace/input-box";
import { useChat } from "@/hooks/use-chat";

export default function ChatPage() {
  const params = useParams();
  const threadId = params.threadId as string;
  const { messages, isLoading, sendMessage } = useChat(threadId);

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-hidden">
        <div className="flex size-full flex-col">
          <div className="flex size-full flex-col overflow-auto">
            <MessageList 
              messages={messages} 
              isLoading={isLoading} 
            />
          </div>
        </div>
      </div>
      <div className="p-4">
        <div className="max-w-2xl mx-auto">
          <InputBox onSubmit={sendMessage} />
        </div>
      </div>
    </div>
  );
}
