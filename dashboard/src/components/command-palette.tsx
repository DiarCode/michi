import { useEffect, useState, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import { useQuery } from "@tanstack/react-query"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  DashboardCircleIcon,
  MapPinIcon,
  Alert01Icon,
  ChartIcon,
  Analytics01Icon,
  Settings01Icon,
  PlayIcon,
  ArrowRight01Icon,
  Sun01Icon,
  Moon02Icon,
  Bus01Icon,
} from "@hugeicons/core-free-icons"

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { fetchStations, fetchAlerts, fetchRoutes } from "@/lib/api"

interface CommandPaletteProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const NAV = [
  {
    label: "Command Center",
    href: "/",
    icon: DashboardCircleIcon,
    shortcut: "G C",
  },
  { label: "Live Map", href: "/map", icon: MapPinIcon, shortcut: "G M" },
  { label: "Alerts", href: "/alerts", icon: Alert01Icon, shortcut: "G A" },
  { label: "Simulation", href: "/simulation", icon: PlayIcon, shortcut: "G S" },
  { label: "Forecast", href: "/forecast", icon: ChartIcon, shortcut: "G F" },
  {
    label: "Executive",
    href: "/executive",
    icon: Analytics01Icon,
    shortcut: "G E",
  },
  {
    label: "Settings",
    href: "/settings",
    icon: Settings01Icon,
    shortcut: "G ,",
  },
] as const

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const navigate = useNavigate()
  const [query, setQuery] = useState("")

  // Fetch real data for search
  const { data: stationsData } = useQuery({
    queryKey: ["stations-search"],
    queryFn: () => fetchStations(),
    enabled: open,
    staleTime: 60_000,
  })

  const { data: routesData } = useQuery({
    queryKey: ["routes-search"],
    queryFn: fetchRoutes,
    enabled: open,
    staleTime: 60_000,
  })

  const { data: alertsData } = useQuery({
    queryKey: ["alerts-search"],
    queryFn: () => fetchAlerts(),
    enabled: open,
    staleTime: 30_000,
  })

  const stations = stationsData?.stations ?? []
  const routes = routesData?.routes ?? []
  const alerts = alertsData?.alerts ?? []

  const filteredStations = useMemo(() => {
    if (!query.trim()) return stations.slice(0, 8)
    const q = query.toLowerCase()
    return stations
      .filter((s) => s.name.toLowerCase().includes(q) || s.district?.toLowerCase().includes(q))
      .slice(0, 8)
  }, [query, stations])

  const filteredRoutes = useMemo(() => {
    if (!query.trim()) return routes.slice(0, 5)
    const q = query.toLowerCase()
    return routes
      .filter((r: { name: string }) => r.name.toLowerCase().includes(q))
      .slice(0, 5)
  }, [query, routes])

  const filteredAlerts = useMemo(() => {
    if (!query.trim()) return alerts.slice(0, 5)
    const q = query.toLowerCase()
    return alerts
      .filter((a) => a.title.toLowerCase().includes(q))
      .slice(0, 5)
  }, [query, alerts])

  useEffect(() => {
    if (!open) setQuery("")
  }, [open])

  function go(href: string) {
    navigate(href)
    onOpenChange(false)
  }

  function toggleTheme() {
    const root = document.documentElement
    root.classList.toggle("dark")
    onOpenChange(false)
  }

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        value={query}
        onValueChange={setQuery}
        placeholder="Search routes, stations, alerts…"
      />
      <CommandList>
        <CommandEmpty>No results for &ldquo;{query}&rdquo;.</CommandEmpty>

        <CommandGroup heading="Navigate">
          {NAV.map((item) => (
            <CommandItem
              key={item.href}
              value={item.label}
              onSelect={() => go(item.href)}
            >
              <HugeiconsIcon icon={item.icon} strokeWidth={1.5} />
              <span>{item.label}</span>
              <HugeiconsIcon
                icon={ArrowRight01Icon}
                strokeWidth={1.5}
                className="ml-auto size-3.5 text-muted-foreground"
              />
            </CommandItem>
          ))}
        </CommandGroup>

        {filteredStations.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Stations">
              {filteredStations.map((s) => (
                <CommandItem
                  key={s.id}
                  value={`station-${s.name}`}
                  onSelect={() => go(`/map`)}
                >
                  <HugeiconsIcon icon={MapPinIcon} strokeWidth={1.5} />
                  <span>{s.name}</span>
                  {s.district && (
                    <span className="ml-1 text-xs text-muted-foreground">
                      · {s.district}
                    </span>
                  )}
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}

        {filteredRoutes.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Routes">
              {filteredRoutes.map((r) => (
                <CommandItem
                  key={r.id}
                  value={`route-${r.name}`}
                  onSelect={() => go(`/forecast`)}
                >
                  <HugeiconsIcon icon={Bus01Icon} strokeWidth={1.5} />
                  <span>{r.name}</span>
                  {r.avg_ridership != null && (
                    <span className="ml-1 text-xs text-muted-foreground">
                      · {r.avg_ridership.toLocaleString()} pax/d
                    </span>
                  )}
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}

        {filteredAlerts.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Alerts">
              {filteredAlerts.map((a) => (
                <CommandItem
                  key={a.id}
                  value={`alert-${a.title}`}
                  onSelect={() => go(`/alerts`)}
                >
                  <HugeiconsIcon icon={Alert01Icon} strokeWidth={1.5} />
                  <span className="truncate">{a.title}</span>
                  <span className="ml-auto text-xs text-muted-foreground">
                    {a.severity}
                  </span>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}

        <CommandSeparator />
        <CommandGroup heading="Quick actions">
          <CommandItem value="toggle-theme" onSelect={toggleTheme}>
            <HugeiconsIcon
              icon={
                typeof window !== "undefined" &&
                document.documentElement.classList.contains("dark")
                  ? Sun01Icon
                  : Moon02Icon
              }
              strokeWidth={1.5}
            />
            Toggle dark mode
          </CommandItem>
          <CommandItem value="go-simulation" onSelect={() => go("/simulation")}>
            <HugeiconsIcon icon={PlayIcon} strokeWidth={1.5} />
            Go to Simulation
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}