"use client";

export function AboutSettingsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
       
        <div>
          <h2 className="text-xl font-semibold">MemoBot</h2>
          <p className="text-sm text-muted-foreground">智能记忆助手</p>
        </div>
      </div>
      
      <div className="rounded-lg border p-4">
        <h3 className="font-medium mb-2">关于 MemoBot</h3>
        <p className="text-sm text-muted-foreground leading-relaxed">
          MemoBot 是一款基于 LangChain 的智能记忆助手应用。它能够帮助您管理和利用记忆数据，提供更加个性化的对话体验。
        </p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between rounded-lg border px-4 py-3">
          <span className="text-sm text-muted-foreground">版本</span>
          <span className="text-sm font-medium">0.1.0</span>
        </div>
        <div className="flex items-center justify-between rounded-lg border px-4 py-3">
          <span className="text-sm text-muted-foreground">构建日期</span>
          <span className="text-sm font-medium">2024-01-15</span>
        </div>
        <div className="flex items-center justify-between rounded-lg border px-4 py-3">
          <span className="text-sm text-muted-foreground">技术栈</span>
          <span className="text-sm font-medium">Next.js 16 + React 19</span>
        </div>
      </div>
    </div>
  );
}
