"use client";

import { useCallback, useState } from "react";
import { Send } from "lucide-react";
import { toast } from "sonner";

import { cn } from "@/lib/utils";

interface InputBoxProps {
  className?: string;
  onSubmit?: (message: string) => void;
}

export function InputBox({ className, onSubmit }: InputBoxProps) {
  const [value, setValue] = useState("");

  const handleSubmit = useCallback(() => {
    if (!value.trim()) {
      toast.warning("请输入消息内容");
      return;
    }
    onSubmit?.(value);
    setValue("");
    toast.success("消息已发送");
  }, [value, onSubmit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <div
      className={cn(
        "bg-background/80 backdrop-blur rounded-2xl border border-muted/30 shadow-sm p-4",
        className
      )}
    >
      <div className="flex gap-2">
        <textarea
          className="flex-1 bg-transparent border-0 resize-none focus:outline-none focus:ring-0 text-sm placeholder:text-muted-foreground"
          placeholder="输入消息... (Enter发送, Shift+Enter换行)"
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          onClick={handleSubmit}
          className="flex items-center justify-center w-10 h-10 bg-accent text-accent-foreground rounded-xl hover:bg-accent/80 transition-colors disabled:opacity-50"
          disabled={!value.trim()}
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
