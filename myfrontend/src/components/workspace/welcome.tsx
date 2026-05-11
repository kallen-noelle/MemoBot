"use client";

import { MessageSquare, Sparkles, Zap } from "lucide-react";

const welcomePrompts = [
  "帮我分析一下这个项目的技术架构",
  "如何优化前端性能？",
  "帮我写一段 React 组件代码",
  "解释一下这个算法的时间复杂度",
];

interface WelcomeProps {
  className?: string;
  onPromptClick?: (prompt: string) => void;
}

export function Welcome({ className, onPromptClick }: WelcomeProps) {
  return (
    <div className={`mt-8 ${className}`}>
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent mb-4">
          <Sparkles className="w-8 h-8 text-accent-foreground" />
        </div>
        <h1 className="text-2xl font-serif font-semibold text-primary mb-2">欢迎使用 MemoBot</h1>
        <p className="text-muted-foreground">你的智能记忆助手</p>
      </div>
      <div className="space-y-3">
        <p className="text-sm font-medium text-muted-foreground">快速开始</p>
        <div className="grid gap-2">
          {welcomePrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => onPromptClick?.(prompt)}
              className="flex items-center gap-3 w-full p-3 text-left rounded-xl border border-muted/30 hover:bg-accent hover:border-accent/50 transition-colors text-sm"
            >
              <MessageSquare className="w-4 h-4 text-muted-foreground shrink-0" />
              <span className="truncate">{prompt}</span>
              <Zap className="w-4 h-4 text-muted-foreground shrink-0 ml-auto" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
