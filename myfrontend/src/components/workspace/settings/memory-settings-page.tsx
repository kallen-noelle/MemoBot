"use client";

import { useState } from "react";

import { Switch } from "@/components/ui/switch";

import { SettingsSection } from "./settings-section";

export function MemorySettingsPage() {
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [maxFacts, setMaxFacts] = useState(100);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [injectionEnabled, setInjectionEnabled] = useState(true);
  const [maxInjectionTokens, setMaxInjectionTokens] = useState(2000);

  return (
    <div className="space-y-8">
      <SettingsSection
        title="记忆功能"
        description="管理智能记忆系统的行为"
      >
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">启用记忆</div>
              <p className="text-sm text-muted-foreground">开启后，系统会自动保存和使用对话记忆</p>
            </div>
            <Switch
              checked={memoryEnabled}
              onCheckedChange={setMemoryEnabled}
            />
          </div>
        </div>
      </SettingsSection>

      <SettingsSection
        title="记忆存储"
        description="配置记忆数据的存储参数"
      >
        <div className="space-y-6">
          <div>
            <label className="flex items-center justify-between text-sm font-medium mb-2">
              <span>最大事实数量</span>
              <span className="text-muted-foreground">{maxFacts}</span>
            </label>
            <input
              type="range"
              min="10"
              max="500"
              value={maxFacts}
              onChange={(e) => setMaxFacts(Number(e.target.value))}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div>
            <label className="flex items-center justify-between text-sm font-medium mb-2">
              <span>事实置信度阈值</span>
              <span className="text-muted-foreground">{confidenceThreshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
      </SettingsSection>

      <SettingsSection
        title="记忆注入"
        description="配置记忆如何注入到对话中"
      >
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">启用记忆注入</div>
              <p className="text-sm text-muted-foreground">将相关记忆注入到对话上下文中</p>
            </div>
            <Switch
              checked={injectionEnabled}
              onCheckedChange={setInjectionEnabled}
            />
          </div>

          <div>
            <label className="flex items-center justify-between text-sm font-medium mb-2">
              <span>最大注入 Token 数</span>
              <span className="text-muted-foreground">{maxInjectionTokens}</span>
            </label>
            <input
              type="range"
              min="500"
              max="5000"
              step="100"
              value={maxInjectionTokens}
              onChange={(e) => setMaxInjectionTokens(Number(e.target.value))}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
      </SettingsSection>
    </div>
  );
}
