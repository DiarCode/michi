import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import MapContainer from "@/components/map/MapContainer"
import TimelineBar from "@/components/map/TimelineBar"
import { useStations } from "@/hooks/useStations"
import { useTimeline } from "@/hooks/useTimeline"
import { fetchStationDetail, fetchStations } from "@/lib/api"
import { useBusStore } from "@/stores/busStore"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { HugeiconsIcon } from "@hugeicons/react"
import { Bus01Icon, MapPinIcon } from "@hugeicons/core-free-icons"
import type { BusPosition, StationDetail } from "@/types"

export default function LiveMap() {
  const { data, isLoading } = useStations()
  const stations = data?.stations ?? []
  const busPositions = useBusStore((s) => s.buses)
  const buses: BusPosition[] = useMemo(
    () => Object.values(busPositions) as unknown as BusPosition[],
    [busPositions]
  )

  const [showConfidence, setShowConfidence] = useState(false)
  const [showBuses, setShowBuses] = useState(true)
  const [showStations, setShowStations] = useState(true)
  const currentHour = new Date().getHours()
  const { data: confidenceData } = useQuery({
    queryKey: ["stations", "confidence", currentHour],
    queryFn: () => fetchStations(currentHour),
    enabled: showConfidence,
  })

  const { mode: timelineMode, getStationData: getTimelineStationData } =
    useTimeline()

  const [selectedStation, setSelectedStation] = useState<StationDetail | null>(
    null
  )
  const [loading, setLoading] = useState(false)

  // Filter visible buses and stations based on toggle state
  const visibleBuses = timelineMode === "historical" ? [] : showBuses ? buses : []
  const visibleStations = showStations ? stations : []

  const handleStationClick = async (stationId: string) => {
    setLoading(true)
    try {
      setSelectedStation(await fetchStationDetail(stationId))
    } catch {
      setSelectedStation(null)
    }
    setLoading(false)
  }

  if (isLoading) {
    return (
      <div className="-m-4 h-[calc(100svh-3rem)] animate-pulse bg-slate-100 md:-m-6 md:rounded-4xl" />
    )
  }

  return (
    <div className="-m-4 flex h-[calc(100svh-3rem)] flex-col overflow-hidden bg-slate-100 md:-m-6 md:rounded-4xl">
      {/* Map fills the entire viewport */}
      <div className="relative min-h-0 flex-1">
        <MapContainer
          stations={visibleStations}
          buses={visibleBuses}
          hour={new Date().getHours()}
          onStationClick={handleStationClick}
          showHeatmap={showStations}
          predictions={[]}
          timelineMode={timelineMode}
          getTimelineStationData={getTimelineStationData}
          confidenceStations={confidenceData?.stations}
          showConfidence={showConfidence}
        />

        {/* Filter toggles */}
        <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
          <div className="flex items-center gap-2 rounded-lg bg-white/90 p-2.5 shadow-md backdrop-blur-sm dark:bg-gray-900/90">
            <div className="flex items-center gap-1.5">
              <HugeiconsIcon icon={Bus01Icon} strokeWidth={1.5} className="size-4 text-blue-600" />
              <Label htmlFor="toggle-buses" className="cursor-pointer text-xs font-medium text-gray-700 select-none dark:text-gray-300">
                Buses
              </Label>
            </div>
            <Switch
              id="toggle-buses"
              checked={showBuses}
              onCheckedChange={setShowBuses}
              size="sm"
            />
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-white/90 p-2.5 shadow-md backdrop-blur-sm dark:bg-gray-900/90">
            <div className="flex items-center gap-1.5">
              <HugeiconsIcon icon={MapPinIcon} strokeWidth={1.5} className="size-4 text-emerald-600" />
              <Label htmlFor="toggle-stations" className="cursor-pointer text-xs font-medium text-gray-700 select-none dark:text-gray-300">
                Stations
              </Label>
            </div>
            <Switch
              id="toggle-stations"
              checked={showStations}
              onCheckedChange={setShowStations}
              size="sm"
            />
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-white/90 p-2.5 shadow-md backdrop-blur-sm dark:bg-gray-900/90">
            <label className="cursor-pointer text-xs font-medium text-gray-700 select-none dark:text-gray-300">
              Confidence
            </label>
            <Switch
              checked={showConfidence}
              onCheckedChange={setShowConfidence}
              size="sm"
            />
          </div>
        </div>

        {/* Floating station detail panel */}
        {selectedStation && (
          <div className="absolute top-4 right-4 bottom-4 z-20 w-80 overflow-y-auto rounded-4xl bg-white p-4 shadow-lg ring-1 ring-foreground/5">
            <button
              onClick={() => setSelectedStation(null)}
              aria-label="Close"
              className="absolute top-2 right-2 inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted"
            >
              ✕
            </button>
            <h3 className="pr-8 font-heading text-base font-medium">
              {selectedStation.station.name}
            </h3>
            <p className="text-xs text-muted-foreground">
              {selectedStation.station.district} ·{" "}
              {selectedStation.station.ridership_24h} pax/24h
            </p>

            {timelineMode === "historical" &&
              (() => {
                const td = getTimelineStationData(selectedStation.station.id)
                if (!td) return null
                return (
                  <div className="mt-2 rounded-2xl bg-amber-500/10 p-2 text-xs">
                    <div className="mb-1 font-semibold text-amber-700 dark:text-amber-400">
                      Timeline Data
                    </div>
                    {td.actual !== null && (
                      <div className="text-muted-foreground">
                        Actual:{" "}
                        <span className="font-mono font-bold">
                          {Math.round(td.actual)} pax
                        </span>
                      </div>
                    )}
                    {td.predicted !== null && (
                      <div className="text-purple-600">
                        Predicted:{" "}
                        <span className="font-mono font-bold">
                          {Math.round(td.predicted)} pax
                        </span>
                      </div>
                    )}
                  </div>
                )
              })()}

            {selectedStation.connected_routes.length > 0 && (
              <div className="mt-3">
                <h4 className="text-sm font-medium">Connected Routes</h4>
                <div className="mt-1 flex flex-wrap gap-2">
                  {selectedStation.connected_routes.map((r) => (
                    <span
                      key={r.id}
                      className="rounded-full px-2 py-1 text-xs text-white"
                      style={{ backgroundColor: r.color ?? "#888" }}
                    >
                      {r.name}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {selectedStation.hourly_ridership.length > 0 && (
              <div className="mt-3">
                <h4 className="text-sm font-medium">Hourly Pattern</h4>
                <div className="mt-1 flex h-20 items-end gap-px">
                  {selectedStation.hourly_ridership.map((h) => {
                    const maxR = Math.max(
                      ...selectedStation.hourly_ridership.map(
                        (x) => x.ridership
                      ),
                      1
                    )
                    const pct = (h.ridership / maxR) * 100
                    const isRush =
                      (h.hour >= 7 && h.hour <= 9) ||
                      (h.hour >= 17 && h.hour <= 19)
                    return (
                      <div
                        key={h.hour}
                        className="flex flex-1 flex-col justify-end"
                      >
                        <div
                          className={
                            isRush ? "bg-amber-400" : "bg-muted-foreground/40"
                          }
                          style={{ height: `${pct}%`, minHeight: 2 }}
                        />
                      </div>
                    )
                  })}
                </div>
                <div className="mt-0.5 flex justify-between text-[8px] text-muted-foreground">
                  <span>0</span>
                  <span>6</span>
                  <span>12</span>
                  <span>18</span>
                  <span>23</span>
                </div>
              </div>
            )}

            {selectedStation.forecast.length > 0 && (
              <div className="mt-3">
                <h4 className="text-sm font-medium">Forecast (next 6h)</h4>
                <div className="mt-1 space-y-1">
                  {selectedStation.forecast.slice(0, 6).map((f, i) => (
                    <div key={i} className="flex justify-between text-xs">
                      <span>
                        {new Date(f.timestamp).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                      <span className="font-mono">{f.predicted} pax</span>
                      <span className="text-muted-foreground">
                        {(f.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedStation.alerts.length > 0 && (
              <div className="mt-3">
                <h4 className="text-sm font-medium text-destructive">
                  Active Alerts
                </h4>
                {selectedStation.alerts.map((a, i) => (
                  <div
                    key={i}
                    className="mt-1 rounded-2xl bg-destructive/10 p-2 text-xs"
                  >
                    <span className="font-semibold">{a.severity}</span>:{" "}
                    {a.title}
                  </div>
                ))}
              </div>
            )}
            {loading && (
              <p className="mt-2 text-xs text-muted-foreground">Loading…</p>
            )}
          </div>
        )}
      </div>

      {/* Timeline at the bottom */}
      <TimelineBar />
    </div>
  )
}

/** Named export used by App.tsx router */
export { LiveMap as LiveMapPage }