"use client";

import { useState, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, ChevronRight, Plus, MessageSquare, Settings, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { cn } from "@/lib/utils";
import { Sidebar, SidebarContent } from "@/components/ui/sidebar";
import { SettingsDialog } from "@/components/workspace/settings/settings-dialog";

interface ChatItem {
  id: string;
  title: string;
}

export function WorkspaceSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [chats, setChats] = useState<ChatItem[]>([]);

  useEffect(() => {
    const savedChats = localStorage.getItem("memobot_chats");
    if (savedChats) {
      setChats(JSON.parse(savedChats));
    } else {
      const defaultChats: ChatItem[] = [
        { id: "1", title: "新对话" },
      ];
      setChats(defaultChats);
    }
  }, []);

  const deleteChat = (e: React.MouseEvent, chatId: string) => {
    e.preventDefault();
    e.stopPropagation();

    const savedChats = localStorage.getItem("memobot_chats");
    const updatedChats = savedChats ? JSON.parse(savedChats).filter((c: ChatItem) => c.id !== chatId) : [];
    localStorage.setItem("memobot_chats", JSON.stringify(updatedChats));

    localStorage.removeItem(`memobot_messages_${chatId}`);
    setChats(updatedChats);

    if (pathname === `/workspace/chats/${chatId}`) {
      router.push("/workspace/chats/new");
    }

    toast.success("对话已删除");
  };

  return (
    <>
      <Sidebar className={cn("border-r transition-all duration-300", collapsed ? "w-16" : "w-64")}>
        <div className="h-12 flex items-center justify-between px-3">
          <div className="flex items-center gap-2">
            {collapsed ? (
              <span className="text-primary font-serif font-semibold text-lg">MB</span>
            ) : (
              <span className="text-primary font-serif font-semibold text-lg">MemoBot</span>
            )}
          </div>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded-lg hover:bg-accent transition-colors"
          >
            {collapsed ? (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronLeft className="w-4 h-4 text-muted-foreground" />
            )}
          </button>
        </div>
        <SidebarContent>
          <div className="p-2">
            <Link
              href="/workspace/chats/new"
              className="flex items-center justify-center gap-2 w-full p-3 bg-accent rounded-lg hover:bg-accent/80 transition-colors"
            >
              <Plus className="w-4 h-4" />
              {!collapsed && <span>新对话</span>}
            </Link>
          </div>

          <div className="px-2 py-4">
            {!collapsed && (
              <div className="text-xs text-muted-foreground font-medium px-3 mb-2">
                最近对话
              </div>
            )}
            <div className="space-y-1">
              {chats.map((chat) => (
                <Link
                  key={chat.id}
                  href={`/workspace/chats/${chat.id}`}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-accent transition-colors group",
                    pathname === `/workspace/chats/${chat.id}` && "bg-accent"
                  )}
                  title={collapsed ? chat.title : undefined}
                >
                  <MessageSquare className="w-4 h-4 text-muted-foreground shrink-0" />
                  {!collapsed && (
                    <>
                      <span className="text-sm truncate flex-1">{chat.title}</span>
                      <button
                        onClick={(e) => deleteChat(e, chat.id)}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/10 rounded transition-opacity"
                        title="删除对话"
                      >
                        <Trash2 className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                      </button>
                    </>
                  )}
                </Link>
              ))}
            </div>
          </div>
        </SidebarContent>
        <div className="p-2">
          <button
            className="flex items-center justify-center gap-3 w-full px-3 py-2 rounded-lg hover:bg-accent transition-colors"
            onClick={() => setSettingsOpen(true)}
            title={collapsed ? "设置" : undefined}
          >
            <Settings className="w-4 h-4 text-muted-foreground" />
            {!collapsed && <span className="text-sm">设置</span>}
          </button>
        </div>
      </Sidebar>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  );
}
