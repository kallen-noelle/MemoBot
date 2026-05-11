"use client";

import { Bell } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";

import { SettingsSection } from "./settings-section";

export function NotificationSettingsPage() {
  const [enabled, setEnabled] = useState(false);
  const [permission, setPermission] = useState<"default" | "granted" | "denied">("default");

  const handleRequestPermission = async () => {
    if ("Notification" in window) {
      const result = await Notification.requestPermission();
      setPermission(result);
      if (result === "granted") {
        setEnabled(true);
      }
    }
  };

  const handleTestNotification = () => {
    if (Notification.permission === "granted") {
      new Notification("测试通知", {
        body: "这是一条测试通知",
      });
    }
  };

  const handleEnableNotification = (value: boolean) => {
    setEnabled(value);
  };

  return (
    <SettingsSection
      title="通知"
      description={
        <div className="flex items-center gap-2">
          <div>管理应用通知设置</div>
          <div>
            <Switch
              disabled={permission !== "granted"}
              checked={permission === "granted" && enabled}
              onCheckedChange={handleEnableNotification}
            />
          </div>
        </div>
      }
    >
      <div className="flex flex-col gap-4">
        {permission === "default" && (
          <Button onClick={handleRequestPermission} variant="default">
            <Bell className="mr-2 size-4" />
            请求通知权限
          </Button>
        )}

        {permission === "denied" && (
          <p className="text-muted-foreground rounded-md border border-amber-200 bg-amber-50 p-3 text-sm">
            通知权限已被拒绝。请在浏览器设置中启用通知权限。
          </p>
        )}

        {permission === "granted" && enabled && (
          <div className="flex flex-col gap-4">
            <Button onClick={handleTestNotification} variant="outline">
              <Bell className="mr-2 size-4" />
              发送测试通知
            </Button>
          </div>
        )}
      </div>
    </SettingsSection>
  );
}
