"use client";

import { WorkspaceSidebar } from "@/components/workspace/workspace-sidebar";

export default function WorkspaceLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="flex h-screen overflow-hidden">
      <WorkspaceSidebar />
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}
