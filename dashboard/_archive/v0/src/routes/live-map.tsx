import { useQuery } from "@tanstack/react-query"
import { useEffect, useMemo, useRef, useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { Alert01Icon, Bus01Icon, LayersIcon, MapPinIcon, RefreshIcon, RouteIcon } from "@hugeicons/core-free-icons"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { fetchRoutes, fetchStations, fetchRichAlerts } from "@/lib/api"
import { cn } from "@/lib/utils"

interface Bounds {
  minLat: number
  maxLat: number
  minLon: number
  maxLon: number
}

const ASTANA_BOUNDS: Bounds = { minLat: 50.9, maxLat: 51.35, minLon: 71.2, maxLon: 71.7 }
const W = 720
const H = 540

function project(lat: number, lon: number, bounds: Bounds) {
  const x = ((lon - bounds.minLon) / (bounds.maxLon - bounds.minLon)) * W
  const y = (1 - (lat - bounds.minLat) / (bounds.maxLat - bounds.minLat)) * H
  return { x, y }
}

export function LiveMapPage() {
  const stations = useQuery({ queryKey: ["stations-map"], queryFn: () => fetchStations() })
  const routes = useQuery({ queryKey: ["routes-map"], queryFn: fetchRoutes })
  const alerts = useQuery({ queryKey: ["rich-alerts-map"], queryFn: fetchRichAlerts, refetchInterval: 20_000 })
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  const projectedStations = useMemo(
    () =>
      (stations.data?.stations ?? []).map((s) => ({
        ...s,
        ...(s.lat && s.lon ? project(s.lat, s.lon, ASTANA_BOUNDS) : { x: 0, y: 0 }),
      })),
    [stations.data],
  )

  const criticalCount = (alerts.data?.alerts ?? []).filter((a) => a.severity === "critical").length
  const warningCount = (alerts.data?.alerts ?? []).filter((a) => a.severity === "warning").length

  useEffect(() => {
    const id = setInterval(() => stations.refetch(), 60_000)
    return () => clearInterval(id)
  }, [stations])

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">Live Map</p>
          <h1 className="font-heading text-3xl font-medium tracking-tight">Network at a glance</h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Schematic view of the Astana bus network, stations, routes and live alerts.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="destructive" className="rounded-xl">{criticalCount} critical</Badge>
          <Badge variant="default" className="rounded-xl">{warningCount} warning</Badge>
          <Button variant="outline" size="icon-sm" onClick={() => stations.refetch()}>
            <HugeiconsIcon icon={RefreshIcon} strokeWidth={2} />
          </Button>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card className="overflow-hidden">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardDescription>Geographic view</CardDescription>
                <CardTitle className="text-lg">Astana</CardTitle>
              </div>
              <Button variant="ghost" size="sm" className="rounded-2xl">
                <HugeiconsIcon icon={LayersIcon} strokeWidth={2} />
                <span>Layers</span>
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {stations.isLoading ? (
              <Skeleton className="m-6 h-[520px] w-[calc(100%-3rem)] rounded-2xl" />
            ) : (
              <div
                ref={containerRef}
                className="relative h-[520px] w-full overflow-hidden"
                style={{
                  background:
                    "radial-gradient(circle at 30% 20%, var(--muted) 0%, transparent 60%), radial-gradient(circle at 80% 80%, var(--accent) 0%, transparent 50%), var(--background)",
                }}
              >
                <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full">
                  <defs>
                    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--border)" strokeWidth="0.5" />
                    </pattern>
                  </defs>
                  <rect width={W} height={H} fill="url(#grid)" opacity={0.5} />
                  <path
                    d="M 60 ${H * 0.7} Q ${W * 0.3} ${H * 0.6}, ${W * 0.5} ${H * 0.55} T ${W * 0.85} ${H * 0.5}"
                    fill="none"
                    stroke="var(--chart-2)"
                    strokeWidth="3"
                    strokeOpacity="0.6"
                    strokeLinecap="round"
                  />
                  <path
                    d="M ${W * 0.1} ${H * 0.2} Q ${W * 0.4} ${H * 0.3}, ${W * 0.6} ${H * 0.4} T ${W * 0.95} ${H * 0.35}"
                    fill="none"
                    stroke="var(--chart-3)"
                    strokeWidth="3"
                    strokeOpacity="0.5"
                    strokeLinecap="round"
                  />
                  <path
                    d="M ${W * 0.2} ${H * 0.85} Q ${W * 0.5} ${H * 0.7}, ${W * 0.7} ${H * 0.75}"
                    fill="none"
                    stroke="var(--chart-4)"
                    strokeWidth="2"
                    strokeOpacity="0.4"
                    strokeLinecap="round"
                  />
                </svg>
                {projectedStations.map((s) => {
                  const alert = (alerts.data?.alerts ?? []).find((a) => a.station_id === s.id)
                  const isSelected = selected === s.id
                  return (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => setSelected(s.id)}
                      className="absolute -translate-x-1/2 -translate-y-1/2"
                      style={{ left: `${(s.x / W) * 100}%`, top: `${(s.y / H) * 100}%` }}
                    >
                      <span
                        className={cn(
                          "block size-3 rounded-full ring-2 ring-background transition-all",
                          alert?.severity === "critical"
                            ? "size-4 bg-destructive"
                            : alert?.severity === "warning"
                              ? "size-3.5 bg-chart-3"
                              : "bg-primary",
                          isSelected && "ring-4 ring-ring/40",
                        )}
                      />
                    </button>
                  )
                })}
                {projectedStations.length > 0 && (
                  <span
                    className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
                    style={{
                      left: `${(projectedStations[0].x / W) * 100}%`,
                      top: `${(projectedStations[0].y / H) * 100}%`,
                    }}
                  >
                    <HugeiconsIcon icon={Bus01Icon} strokeWidth={2} className="size-5 text-foreground drop-shadow-md" />
                  </span>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardDescription>Stations</CardDescription>
              <CardTitle className="text-lg">Selected</CardTitle>
            </CardHeader>
            <CardContent>
              {selected ? (
                <SelectedStation
                  stationId={selected}
                  stations={stations.data?.stations ?? []}
                  alerts={alerts.data?.alerts ?? []}
                />
              ) : (
                <p className="text-sm text-muted-foreground">Tap any dot on the map to inspect a station.</p>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardDescription>Routes</CardDescription>
              <CardTitle className="text-lg">Active ({routes.data?.routes.length ?? 0})</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1.5">
                {(routes.data?.routes ?? []).slice(0, 8).map((r) => (
                  <li
                    key={r.id}
                    className="flex items-center gap-2 rounded-2xl border border-border bg-card p-2.5"
                  >
                    <span
                      className="size-2.5 shrink-0 rounded-full"
                      style={{ background: r.color ?? "var(--chart-1)" }}
                    />
                    <HugeiconsIcon icon={RouteIcon} strokeWidth={2} className="size-3.5 text-muted-foreground" />
                    <span className="truncate text-sm font-medium">{r.name}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function SelectedStation({
  stationId,
  stations,
  alerts,
}: {
  stationId: string
  stations: Array<{ id: string; name: string; district?: string }>
  alerts: Array<{ id: number; title: string; severity: string; station_id?: string }>
}) {
  const station = stations.find((s) => s.id === stationId)
  const stationAlerts = alerts.filter((a) => a.station_id === stationId)
  if (!station) return <p className="text-sm text-muted-foreground">Unknown station.</p>
  return (
    <div className="space-y-3">
      <div>
        <p className="text-sm font-medium">{station.name}</p>
        <p className="text-xs text-muted-foreground">{station.district ?? station.id}</p>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <HugeiconsIcon icon={MapPinIcon} strokeWidth={2} className="size-3.5" />
        {station.id}
      </div>
      {stationAlerts.length > 0 ? (
        <ul className="space-y-1.5">
          {stationAlerts.map((a) => (
            <li
              key={a.id}
              className="flex items-center gap-2 rounded-2xl border border-border bg-card p-2 text-xs"
            >
              <HugeiconsIcon
                icon={Alert01Icon}
                strokeWidth={2}
                className={cn(
                  "size-3.5",
                  a.severity === "critical" ? "text-destructive" : "text-chart-3",
                )}
              />
              <span className="truncate">{a.title}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="rounded-2xl border border-dashed border-border p-3 text-xs text-muted-foreground">
          No active alerts at this station.
        </p>
      )}
    </div>
  )
}

export default LiveMapPage
