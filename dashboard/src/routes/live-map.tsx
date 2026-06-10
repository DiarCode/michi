import { useState, useMemo } from "react";
import MapContainer from "@/components/map/MapContainer";
import TimelineBar from "@/components/map/TimelineBar";
import { useStations } from "@/hooks/useStations";
import { useTimeline } from "@/hooks/useTimeline";
import { fetchStationDetail } from "@/lib/api";
import { useBusStore } from "@/stores/busStore";
import type { BusPosition, StationDetail } from "@/types";

export default function LiveMap() {
  const { data, isLoading } = useStations();
  const stations = data?.stations ?? [];
  const busPositions = useBusStore((s) => s.buses);
  const buses: BusPosition[] = useMemo(
    () => Object.values(busPositions) as unknown as BusPosition[],
    [busPositions]
  );

  const { mode: timelineMode, getStationData: getTimelineStationData } = useTimeline();

  const [selectedStation, setSelectedStation] = useState<StationDetail | null>(null);
  const [loading, setLoading] = useState(false);

  // In historical mode, hide buses (they didn't exist in the past)
  const visibleBuses = timelineMode === "historical" ? [] : buses;

  const handleStationClick = async (stationId: string) => {
    setLoading(true);
    try {
      setSelectedStation(await fetchStationDetail(stationId));
    } catch {
      setSelectedStation(null);
    }
    setLoading(false);
  };

  if (isLoading) {
    return (
      <div className="-m-4 md:-m-6 h-[calc(100svh-3rem)] bg-slate-100 animate-pulse md:rounded-4xl" />
    );
  }

  return (
    <div className="-m-4 md:-m-6 h-[calc(100svh-3rem)] flex flex-col bg-slate-100 overflow-hidden md:rounded-4xl">
      {/* Map fills the entire viewport */}
      <div className="relative flex-1 min-h-0">
        <MapContainer
          stations={stations}
          buses={visibleBuses}
          hour={new Date().getHours()}
          onStationClick={handleStationClick}
          showHeatmap
          predictions={[]}
          timelineMode={timelineMode}
          getTimelineStationData={getTimelineStationData}
        />

        {/* Floating station detail panel */}
        {selectedStation && (
          <div className="absolute top-4 right-4 bottom-4 w-80 rounded-4xl bg-white shadow-lg ring-1 ring-foreground/5 overflow-y-auto p-4 z-10">
            <button
              onClick={() => setSelectedStation(null)}
              aria-label="Close"
              className="absolute top-2 right-2 size-7 inline-flex items-center justify-center rounded-full text-muted-foreground hover:bg-muted transition-colors"
            >
              ✕
            </button>
            <h3 className="font-heading text-base font-medium pr-8">
              {selectedStation.station.name}
            </h3>
            <p className="text-xs text-muted-foreground">
              {selectedStation.station.district} · {selectedStation.station.ridership_24h} pax/24h
            </p>

            {timelineMode === "historical" && (() => {
              const td = getTimelineStationData(selectedStation.station.id);
              if (!td) return null;
              return (
                <div className="mt-2 p-2 rounded-2xl bg-amber-500/10 text-xs">
                  <div className="font-semibold text-amber-700 dark:text-amber-400 mb-1">Timeline Data</div>
                  {td.actual !== null && (
                    <div className="text-muted-foreground">
                      Actual: <span className="font-mono font-bold">{Math.round(td.actual)} pax</span>
                    </div>
                  )}
                  {td.predicted !== null && (
                    <div className="text-purple-600">
                      Predicted: <span className="font-mono font-bold">{Math.round(td.predicted)} pax</span>
                    </div>
                  )}
                </div>
              );
            })()}

            {selectedStation.connected_routes.length > 0 && (
              <div className="mt-3">
                <h4 className="font-medium text-sm">Connected Routes</h4>
                <div className="flex gap-2 mt-1 flex-wrap">
                  {selectedStation.connected_routes.map((r) => (
                    <span
                      key={r.id}
                      className="px-2 py-1 text-xs rounded-full text-white"
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
                <h4 className="font-medium text-sm">Hourly Pattern</h4>
                <div className="flex items-end gap-px h-20 mt-1">
                  {selectedStation.hourly_ridership.map((h) => {
                    const maxR = Math.max(...selectedStation.hourly_ridership.map((x) => x.ridership), 1);
                    const pct = (h.ridership / maxR) * 100;
                    const isRush = (h.hour >= 7 && h.hour <= 9) || (h.hour >= 17 && h.hour <= 19);
                    return (
                      <div key={h.hour} className="flex-1 flex flex-col justify-end">
                        <div
                          className={isRush ? "bg-amber-400" : "bg-muted-foreground/40"}
                          style={{ height: `${pct}%`, minHeight: 2 }}
                        />
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[8px] text-muted-foreground mt-0.5">
                  <span>0</span><span>6</span><span>12</span><span>18</span><span>23</span>
                </div>
              </div>
            )}

            {selectedStation.forecast.length > 0 && (
              <div className="mt-3">
                <h4 className="font-medium text-sm">Forecast (next 6h)</h4>
                <div className="space-y-1 mt-1">
                  {selectedStation.forecast.slice(0, 6).map((f, i) => (
                    <div key={i} className="flex justify-between text-xs">
                      <span>{new Date(f.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                      <span className="font-mono">{f.predicted} pax</span>
                      <span className="text-muted-foreground">{(f.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedStation.alerts.length > 0 && (
              <div className="mt-3">
                <h4 className="font-medium text-sm text-destructive">Active Alerts</h4>
                {selectedStation.alerts.map((a, i) => (
                  <div key={i} className="mt-1 p-2 rounded-2xl bg-destructive/10 text-xs">
                    <span className="font-semibold">{a.severity}</span>: {a.title}
                  </div>
                ))}
              </div>
            )}
            {loading && <p className="text-xs text-muted-foreground mt-2">Loading…</p>}
          </div>
        )}
      </div>

      {/* Timeline at the bottom */}
      <TimelineBar />
    </div>
  );
}

/** Named export used by App.tsx router */
export { LiveMap as LiveMapPage };
