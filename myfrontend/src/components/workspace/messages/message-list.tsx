"use client";

import { useEffect, useRef, useState } from "react";
import { Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

export interface Message {
  id: string;
  type: "human" | "assistant";
  content: string;
  timestamp: Date;
}

interface MessageListProps {
  className?: string;
  messages?: Message[];
  isLoading?: boolean;
}

export function MessageList({
  className,
  messages,
  isLoading = false
}: MessageListProps) {
  const defaultMessages: Message[] = [
    {
      id: "1",
      type: "human",
      content: "你好！我想了解一下这个项目的架构设计。",
      timestamp: new Date(Date.now() - 3600000),
    },
    {
      id: "2",
      type: "assistant",
      content: "你好！这个项目使用分层架构设计，主要分为前端层、API层和数据层。前端使用 React + Next.js 构建，后端采用 Node.js + Express，数据库使用 PostgreSQL。",
      timestamp: new Date(Date.now() - 3590000),
    },
  ];

  const displayMessages = messages || defaultMessages;
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayMessages.length, isLoading]);

  const copyToClipboard = async (text: string): Promise<boolean> => {
    try {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.cssText = "position:fixed;left:-9999px;top:-9999px;opacity:0;";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();

      let success = false;
      try {
        success = document.execCommand("copy");
      } catch {
        success = false;
      }

      document.body.removeChild(textarea);
      return success;
    } catch {
      return false;
    }
  };

  const copyMessage = async (content: string) => {
    const success = await copyToClipboard(content);
    if (success) {
      toast.success("已复制到剪贴板");
    } else {
      toast.error("复制失败，请手动选择文本复制");
    }
  };

  return (
    <div className={cn("flex size-full flex-col", className)}>
      <div className="mx-auto w-full max-w-3xl gap-12 pt-12 pb-4 px-4">
        {displayMessages.map((message) => (
          <div
            key={message.id}
            className={cn(
              "flex w-full gap-4",
              message.type === "human" && "justify-end"
            )}
          >
            {message.type === "assistant" && (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-bold">
                M
              </div>
            )}
            <div
              className={cn(
                "max-w-[80%] rounded-2xl px-4 py-3",
                message.type === "human"
                  ? "bg-accent text-accent-foreground"
                  : "bg-muted"
              )}
            >
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
              <div className="mt-2 flex items-center justify-between">
                <p className="text-xs text-muted-foreground opacity-70">
                  {message.timestamp.toLocaleTimeString("zh-CN")}
                </p>
                {message.type === "assistant" && (
                  <CopyButton content={message.content} />
                )}
              </div>
            </div>
            {message.type === "human" && (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent">
                👤
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="flex w-full gap-4">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent">
              🤖
            </div>
            <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-muted">
              <div className="flex gap-1">
                <div className="w-2 h-2 rounded-full bg-muted-foreground/30 animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-2 h-2 rounded-full bg-muted-foreground/30 animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-2 h-2 rounded-full bg-muted-foreground/30 animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="h-40" />
    </div>
  );
}

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async (text: string): Promise<boolean> => {
    try {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.cssText = "position:fixed;left:-9999px;top:-9999px;opacity:0;";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();

      let success = false;
      try {
        success = document.execCommand("copy");
      } catch {
        success = false;
      }

      document.body.removeChild(textarea);
      return success;
    } catch {
      return false;
    }
  };

  const handleCopy = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const success = await copyToClipboard(content);
    if (success) {
      setCopied(true);
      toast.success("已复制到剪贴板");
      setTimeout(() => setCopied(false), 2000);
    } else {
      toast.error("复制失败，请手动选择文本复制");
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="p-1 hover:bg-accent/50 rounded transition-colors"
      title="复制"
    >
      {copied ? (
        <Check className="w-3 h-3 text-green-500" />
      ) : (
        <Copy className="w-3 h-3 text-muted-foreground" />
      )}
    </button>
  );
}
