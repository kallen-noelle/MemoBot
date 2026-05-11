"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { MessageList } from "@/components/workspace/messages/message-list";
import { InputBox } from "@/components/workspace/input-box";
import { Welcome } from "@/components/workspace/welcome";
import { useChat } from "@/hooks/use-chat";

export default function NewChatPage() {
  const router = useRouter();
  const { messages, isLoading, chatId, isNewChat, sendMessage } = useChat();

  useEffect(() => {
    if (!isNewChat && chatId !== "new") {
      router.replace(`/workspace/chats/${chatId}`);
    }
  }, [isNewChat, chatId, router]);

  const handleSubmit = (message: string) => {
    sendMessage(message);
  };

  const handlePromptClick = (prompt: string) => {
    sendMessage(prompt);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-hidden">
        <div className="flex size-full flex-col">
          <div className="flex size-full flex-col overflow-auto">
            {messages.length === 0 && (
              <Welcome className="mx-auto max-w-md w-full px-4" onPromptClick={handlePromptClick} />
            )}
            <MessageList messages={messages} isLoading={isLoading} />
          </div>
        </div>
      </div>
      <div className="p-4">
        <div className="max-w-2xl mx-auto">
          <InputBox onSubmit={handleSubmit} />
        </div>
      </div>
    </div>
  );
}
