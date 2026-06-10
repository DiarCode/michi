import * as React from "react"
import { NavLink } from "react-router-dom"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Alert01Icon,
  Analytics01Icon,
  ChartIcon,
  DashboardCircleIcon,
  MapPinIcon,
  PlayIcon,
  Settings01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Toaster } from "@/components/ui/sonner"
import { ConnectionIndicator } from "@/components/connection-indicator"
import { MichiLogo } from "@/components/michi-logo"
import { RoleSwitcher } from "@/components/role-switcher"
import { ThemeToggle } from "@/components/theme-toggle"
import { CommandPalette } from "@/components/command-palette"
import { useRoleStore, ROLE_LABELS } from "@/stores/role-store"
import { cn } from "@/lib/utils"

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
  const role = useRoleStore((s) => s.role)
  return (
    <Sidebar variant="sidebar" collapsible="offcanvas">
      <SidebarHeader>
        <div className="flex items-center justify-between px-1 py-1">
          <MichiLogo />
          <SidebarTrigger className="md:hidden" />
        </div>
        <div className="px-1 pt-1 text-xs text-muted-foreground">
          Astana Bus Network Intelligence
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Workspace</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV.map((item) => {
                const Icon = item.icon
                return (
                  <SidebarMenuItem key={item.to}>
                    <SidebarMenuButton
                      render={
                        <NavLink to={item.to} end={item.end as boolean} />
                      }
                    >
                      <HugeiconsIcon icon={Icon} strokeWidth={2} />
                      <span>{item.label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <div className="flex flex-col gap-2 px-1">
          <ConnectionIndicator />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Role: {ROLE_LABELS[role]}</span>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}

function AppTopbar({ onOpenPalette }: { onOpenPalette: () => void }) {
  const { isMobile } = useSidebar()
  return (
    <header
      className={cn(
        "sticky top-0 z-30 flex h-16 items-center gap-2 border-b border-border bg-background/70 px-4 backdrop-blur-md",
        "md:px-6",
      )}
    >
      <div className="flex items-center gap-2">
        <SidebarTrigger />
        {isMobile && <MichiLogo />}
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={onOpenPalette}
        className="ml-2 hidden w-72 justify-start gap-2 rounded-2xl border-dashed text-muted-foreground sm:flex"
      >
        <HugeiconsIcon icon={Search01Icon} strokeWidth={2} />
        <span>Search…</span>
        <span className="ml-auto text-xs tracking-widest opacity-60">⌘ K</span>
      </Button>
      <div className="ml-auto flex items-center gap-2">
        <RoleSwitcher />
        <ThemeToggle />
      </div>
    </header>
  )
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false)

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setOpen((v) => !v)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  return (
    <TooltipProvider delay={200}>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
          <AppTopbar onOpenPalette={() => setOpen(true)} />
          <main className="flex-1 px-4 py-6 md:px-8 md:py-10">{children}</main>
        </SidebarInset>
        <CommandPalette open={open} onOpenChange={setOpen} />
        <Toaster />
      </SidebarProvider>
    </TooltipProvider>
  )
}
