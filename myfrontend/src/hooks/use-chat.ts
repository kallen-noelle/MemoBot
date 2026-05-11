"use client";

import { useState, useCallback, useEffect } from "react";
import { toast } from "sonner";

import type { Message } from "@/components/workspace/messages/message-list";
import { api, ChatRequest, ChatResponse } from "@/lib/api";

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  isError: boolean;
}

export function useChat(threadId?: string) {
  const [chatId, setChatId] = useState<string>(threadId || "new");
  const [isNewChat, setIsNewChat] = useState<boolean>(!threadId);

  const loadMessages = useCallback((id: string): Message[] => {
    if (!id || id === "new") return [];
    const saved = localStorage.getItem(`memobot_messages_${id}`);
    if (saved) {
      const messages = JSON.parse(saved);
      return messages.map((m: Message) => ({
        ...m,
        timestamp: new Date(m.timestamp),
      }));
    }
    return [];
  }, []);

  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    isError: false,
  });

  useEffect(() => {
    if (chatId && chatId !== "new") {
      setState(prev => ({
        ...prev,
        messages: loadMessages(chatId),
      }));
    }
  }, [chatId, loadMessages]);

  const saveMessages = useCallback((messages: Message[], id: string) => {
    if (id && id !== "new") {
      localStorage.setItem(`memobot_messages_${id}`, JSON.stringify(messages));
    }
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    let currentChatId = chatId;

    if (state.messages.length === 0) {
      if (isNewChat) {
        currentChatId = Date.now().toString();
        setChatId(currentChatId);
        setIsNewChat(false);
      }

      const title = content.slice(0, 6);
      const savedChats = localStorage.getItem("memobot_chats");
      const chats = savedChats ? JSON.parse(savedChats) : [];
      chats.unshift({ id: currentChatId, title });
      localStorage.setItem("memobot_chats", JSON.stringify(chats.slice(0, 10)));
    }

    const newMessage: Message = {
      id: Date.now().toString(),
      type: "human",
      content,
      timestamp: new Date(),
    };

    setState((prev) => {
      const newMessages = [...prev.messages, newMessage];
      if (currentChatId !== "new") {
        saveMessages(newMessages, currentChatId);
      }
      return { ...prev, messages: newMessages, isLoading: true, isError: false };
    });

    try {
      const request: ChatRequest = {
        uid: "test_user_123",
        thread_id: threadId || "default_thread",
        message: content,
        model_name: "qwen-plus",
      };

      const fetchOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      };
      const response = await fetch(api.chat.send, fetchOptions);

      console.log("响应状态:", response.status, response.statusText);

      const responseText = await response.text();
      console.log("响应原始:", responseText);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = JSON.parse(responseText);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: data.message,
        timestamp: new Date(),
      };

      setState((prev) => {
        const newMessages = [...prev.messages, assistantMessage];
        if (currentChatId !== "new") {
          saveMessages(newMessages, currentChatId);
        }
        return { ...prev, messages: newMessages, isLoading: false };
      });

      toast.success("消息已发送");
    } catch (error) {
      console.error("发送消息失败:", error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
        isError: true,
      }));
      toast.error("发送消息失败，请重试");
    }
  }, [chatId, isNewChat, state.messages.length, threadId, saveMessages]);

  const clearMessages = useCallback(() => {
    setState((prev) => ({
      ...prev,
      messages: [],
    }));
    if (chatId && chatId !== "new") {
      localStorage.removeItem(`memobot_messages_${chatId}`);
    }
  }, [chatId]);

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    isError: state.isError,
    chatId,
    isNewChat,
    sendMessage,
    clearMessages,
  };
}
