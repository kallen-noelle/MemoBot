"use client";

import { useState } from "react";

import { Switch } from "@/components/ui/switch";

import { SettingsSection } from "./settings-section";

const tools = [
  { id: "web_search", name: "网页搜索", description: "允许 Agent 进行网页搜索", enabled: true },
  { id: "calculator", name: "计算器", description: "允许 Agent 进行数学计算", enabled: true },
  { id: "file_reader", name: "文件读取", description: "允许 Agent 读取上传的文件", enabled: true },
  { id: "code_execution", name: "代码执行", description: "允许 Agent 执行代码", enabled: false },
];

export function ToolSettingsPage() {
  const [toolList, setToolList] = useState(tools);

  const toggleTool = (id: string) => {
    setToolList(prev =>
      prev.map(tool =>
        tool.id === id ? { ...tool, enabled: !tool.enabled } : tool
      )
    );
  };

  return (
    <SettingsSection
      title="工具"
      description="管理 Agent 可用的工具"
    >
      <div className="space-y-2">
        {toolList.map((tool) => (
          <div
            key={tool.id}
            className="flex items-center justify-between rounded-lg border p-4"
          >
            <div>
              <div className="font-medium">{tool.name}</div>
              <p className="text-sm text-muted-foreground">{tool.description}</p>
            </div>
            <Switch
              checked={tool.enabled}
              onCheckedChange={() => toggleTool(tool.id)}
            />
          </div>
        ))}
      </div>
    </SettingsSection>
  );
}
