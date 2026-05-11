"use client";

import { Sparkles } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";

import { SettingsSection } from "./settings-section";

const skills = [
  { id: "pdf_processing", name: "PDF 处理", description: "提取和分析 PDF 内容", license: "MIT", category: "public", enabled: true },
  { id: "data_analysis", name: "数据分析", description: "对数据进行统计分析", license: "Apache", category: "public", enabled: true },
  { id: "text_summarization", name: "文本摘要", description: "生成文本摘要", license: "MIT", category: "custom", enabled: false },
];

export function SkillSettingsPage({ onClose }: { onClose?: () => void }) {
  const [skillList, setSkillList] = useState(skills);

  const toggleSkill = (id: string) => {
    setSkillList(prev =>
      prev.map(skill =>
        skill.id === id ? { ...skill, enabled: !skill.enabled } : skill
      )
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <SettingsSection
          title="技能管理"
          description="管理和配置可用技能"
        />
        <Button variant="outline" size="sm">
          <Sparkles className="mr-2 size-4" />
          安装技能
        </Button>
      </div>

      <div className="space-y-2">
        {skillList.map((skill) => (
          <div
            key={skill.id}
            className="flex items-center justify-between rounded-lg border p-4"
          >
            <div className="flex-1">
              <div className="font-medium">{skill.name}</div>
              <p className="text-sm text-muted-foreground">{skill.description}</p>
              <div className="flex items-center gap-4 mt-2">
                <span className="text-xs text-muted-foreground">许可证: {skill.license}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${skill.category === "public" ? "bg-blue-100 text-blue-800" : "bg-green-100 text-green-800"}`}>
                  {skill.category === "public" ? "公共" : "自定义"}
                </span>
              </div>
            </div>
            <Switch
              checked={skill.enabled}
              onCheckedChange={() => toggleSkill(skill.id)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
