"use client";

import { cn } from "@/lib/utils";

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Sidebar({ children, className, ...props }: SidebarProps) {
  return (
    <div
      className={cn(
        "flex size-full flex-col bg-sidebar border-sidebar-border rounded-r-xl",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

interface SidebarContentProps extends React.HTMLAttributes<HTMLDivElement> {}

export function SidebarContent({ children, className, ...props }: SidebarContentProps) {
  return (
    <div className={cn("flex-1 overflow-auto", className)} {...props}>
      {children}
    </div>
  );
}
