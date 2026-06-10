import * as React from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Alert01Icon,
  Analytics01Icon,
  ChartIcon,
  DashboardCircleIcon,
  MapPinIcon,
  PlayIcon,
  Settings01Icon,
  SourceCodeSquareIcon,
} from "@hugeicons/core-free-icons"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Toaster } from "@/components/ui/sonner"
import { ConnectionIndicator } from "@/components/connection-indicator"
import { MichiLogo } from "@/components/michi-logo"
import { RoleSwitcher } from "@/components/role-switcher"
import { CommandPalette } from "@/components/command-palette"
import { Kbd } from "@/components/ui/kbd"

const NAV = [
  { to: "/", label: "Command Center", icon: DashboardCircleIcon, end: true },
  { to: "/map", label: "Live Map", icon: MapPinIcon, end: false },
  { to: "/alerts", label: "Alerts", icon: Alert01Icon, end: false },
  { to: "/simulation", label: "Simulation", icon: PlayIcon, end: false },
  { to: "/forecast", label: "Forecast", icon: ChartIcon, end: false },
  { to: "/executive", label: "Executive", icon: Analytics01Icon, end: false },
  { to: "/settings", label: "Settings", icon: Settings01Icon, end: false },
] as const

function AppSidebar() {
  return (
    <Sidebar variant="sidebar" collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center justify-between px-1 py-1">
          <MichiLogo />
          <SidebarTrigger className="md:hidden" />
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV.map((item) => (
                <SidebarMenuItem key={item.to}>
                  <SidebarMenuButton asChild tooltip={item.label}>
                    <a href={item.to} aria-current={item.end ? "page" : undefined}>
                      <HugeiconsIcon icon={item.icon} strokeWidth={1.5} />
                      <span>{item.label}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <ConnectionIndicator />
        <div className="flex items-center justify-between gap-1">
          <RoleSwitcher />
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false)

  React.useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setOpen((v) => !v)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  return (
    <TooltipProvider delayDuration={200}>
      <SidebarProvider defaultOpen>
        <AppSidebar />
        <SidebarInset className="bg-background">
          <header className="sticky top-0 z-30 flex h-12 items-center justify-between gap-2 border-b border-border/60 bg-background/80 px-4 backdrop-blur-md">
            <div className="flex items-center gap-2">
              <SidebarTrigger className="size-7" />
              <button
                onClick={() => setOpen(true)}
                className="flex items-center gap-2 rounded-2xl border border-border/60 bg-muted/40 px-3 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted"
              >
                <HugeiconsIcon icon={SourceCodeSquareIcon} strokeWidth={1.5} className="size-3.5" />
                <span>Search Michi</span>
                <Kbd>⌘</Kbd>
                <Kbd>K</Kbd>
              </button>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="hidden md:inline">Astana · Live</span>
            </div>
          </header>
          <main className="min-h-[calc(100svh-3rem)] p-4 md:p-6">{children}</main>
        </SidebarInset>
        <CommandPalette open={open} onOpenChange={setOpen} />
        <Toaster richColors position="bottom-right" />
      </SidebarProvider>
    </TooltipProvider>
  )
}
