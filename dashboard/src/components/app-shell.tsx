import * as React from "react"
import { Link, useLocation } from "react-router-dom"
import { HugeiconsIcon } from "@hugeicons/react"
import { useQuery } from "@tanstack/react-query"
import {
  Alert01Icon,
  Analytics01Icon,
  ChartIcon,
  DashboardCircleIcon,
  MapPinIcon,
  PlayIcon,
  Settings01Icon,
  SourceCodeSquareIcon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons"
import { format } from "date-fns"

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
import { CommandPalette } from "@/components/command-palette"
import { Kbd } from "@/components/ui/kbd"
import { fetchWeatherCurrent } from "@/lib/api"
/** Map WMO weather codes to emoji icons. */
function weatherEmoji(code: number | null): string {
  if (code == null) return "⚫"
  if (code === 0) return "☀️"
  if (code === 1) return "🌤️"
  if (code === 2) return "⛅"
  if (code === 3) return "☁️"
  if (code >= 45 && code <= 48) return "🌫️"
  if (code >= 51 && code <= 55) return "🌦️"
  if (code >= 61 && code <= 65) return "🌧️"
  if (code >= 71 && code <= 75) return "🌨️"
  if (code >= 80 && code <= 82) return "🌧️"
  if (code >= 95) return "⛈️"
  return "⚫"
}

function WeatherIndicator() {
  const { data: weather } = useQuery({
    queryKey: ["weather-current"],
    queryFn: fetchWeatherCurrent,
    refetchInterval: 30 * 60 * 1000,
    retry: 1,
    staleTime: 15 * 60 * 1000,
  })

  if (!weather || weather.temperature_c == null) return null

  const temp = Math.round(weather.temperature_c)
  const emoji = weatherEmoji(weather.weather_code)
  const desc = weather.description ?? ""
  const formattedDate = format(new Date(), "EEE, MMM d")

  return (
    <span
      className="hidden items-center gap-1.5 md:inline-flex"
      title={desc}
    >
      {emoji} {temp}°C · {formattedDate}
    </span>
  )
}

const NAV = [
  { to: "/", label: "Command Center", icon: DashboardCircleIcon, end: true },
  { to: "/map", label: "Live Map", icon: MapPinIcon, end: false },
  { to: "/alerts", label: "Alerts", icon: Alert01Icon, end: false },
  { to: "/simulation", label: "Simulation", icon: PlayIcon, end: false },
  { to: "/forecast", label: "Forecast", icon: ChartIcon, end: false },
  { to: "/crowding", label: "Crowding", icon: UserGroupIcon, end: false },
  { to: "/executive", label: "Executive", icon: Analytics01Icon, end: false },
  { to: "/settings", label: "Settings", icon: Settings01Icon, end: false },
] as const

function AppSidebar() {
  const location = useLocation()

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
              {NAV.map((item) => {
                const isActive = item.end
                  ? location.pathname === item.to
                  : location.pathname.startsWith(item.to)
                return (
                  <SidebarMenuItem key={item.to}>
                    <SidebarMenuButton
                      asChild
                      tooltip={item.label}
                      isActive={isActive}
                    >
                      <Link to={item.to}>
                        <HugeiconsIcon icon={item.icon} strokeWidth={1.5} />
                        <span>{item.label}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <ConnectionIndicator />
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
                <HugeiconsIcon
                  icon={SourceCodeSquareIcon}
                  strokeWidth={1.5}
                  className="size-3.5"
                />
                <span>Search Michi</span>
                <Kbd>⌘</Kbd>
                <Kbd>K</Kbd>
              </button>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <WeatherIndicator />
              <span className="hidden md:inline">Astana · Live</span>
            </div>
          </header>
          <main className="min-h-[calc(100svh-3rem)] p-4 md:p-6">
            {children}
          </main>
        </SidebarInset>
        <CommandPalette open={open} onOpenChange={setOpen} />
        <Toaster richColors position="bottom-right" />
      </SidebarProvider>
    </TooltipProvider>
  )
}