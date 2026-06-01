import { useState, useMemo } from "react";
import MapContainer from "@/components/map/MapContainer";
import TimelineBar from "@/components/map/TimelineBar";
import { useStations } from "@/hooks/useStations";
import { useTimeline } from "@/hooks/useTimeline";
import { fetchStationDetail } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { useBusStore } from "@/stores/busStore";
import type { BusPosition, StationDetail } from "@/types";

export default function LiveMap() {
  const { mode: timelineMode, currentTime, getStationData: getTimelineStationData } = useTimeline();

  // Derive hour from timeline position so station colors/heatmap update when scrubber moves
  const timelineHour = useMemo(() => {
    if (timelineMode === "historical" && currentTime) {
      return new Date(currentTime).getHours();
    }
    return new Date().getHours();
  }, [timelineMode, currentTime]);

  const { data, isLoading } = useStations(timelineHour);
  const stations = data?.stations ?? [];
  const busPositions = useBusStore((s) => s.buses);
  const buses: BusPosition[] = useMemo(() => Object.values(busPositions) as unknown as BusPosition[], [busPositions]);

  const [selectedStation, setSelectedStation] = useState<StationDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);

  const handleStationClick = async (stationId: string) => {
    setLoading(true);
    try { setSelectedStation(await fetchStationDetail(stationId)); }
    catch (err: any) { showToast.error(`Failed to load station: ${err.message}`); setSelectedStation(null); }
    setLoading(false);
  };

  // In historical mode, hide buses; otherwise show all
  const visibleBuses = timelineMode === "historical" ? [] : buses;

  if (isLoading) {
    return <div className="h-[calc(100vh-4rem)] bg-gray-100 dark:bg-gray-800 animate-pulse" />;
  }

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      {/* Full-width map area */}
      <div className="flex-1 relative">
        <MapContainer
          stations={stations}
          buses={visibleBuses}
          hour={timelineHour}
          onStationClick={handleStationClick}
          showHeatmap={showHeatmap}
          timelineMode={timelineMode}
          getTimelineStationData={getTimelineStationData}
        />

        {/* Floating heatmap toggle (top-right) */}
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="absolute top-4 right-4 z-10 px-3 py-1.5 text-xs font-medium rounded-lg shadow-md transition-colors bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm text-gray-700 dark:text-gray-300 hover:bg-white dark:hover:bg-gray-900"
        >
          {showHeatmap ? "● Heatmap" : "○ Heatmap"}
        </button>

        {/* Floating bus count badge (top-left) */}
        {visibleBuses.length > 0 && (
          <div className="absolute top-4 left-4 z-10 px-3 py-1.5 text-xs font-medium rounded-lg shadow-md bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm text-gray-700 dark:text-gray-300">
            {visibleBuses.length} buses active · {String(timelineHour).padStart(2, "0")}:00
          </div>
        )}

        {/* Station detail overlay (right panel, only when a station is clicked) */}
        {selectedStation && (
          <div className="absolute top-0 right-0 w-80 h-full bg-white/95 dark:bg-gray-900/95 shadow-lg overflow-y-auto p-4">
            <button onClick={() => setSelectedStation(null)} className="absolute top-2 right-2 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 text-lg">✕</button>
            <h3 className="font-bold text-lg dark:text-white">{selectedStation.station.name}</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">{selectedStation.station.district} · {selectedStation.station.ridership_24h?.toLocaleString() ?? "—"} passengers per day</p>

            {timelineMode === "historical" && (() => {
              const td = getTimelineStationData(selectedStation.station.id);
              if (!td) return null;
              return (
                <div className="mt-2 p-2 bg-amber-50 dark:bg-amber-900/30 rounded text-xs">
                  <div className="font-semibold text-amber-700 dark:text-amber-300 mb-1">Timeline Data</div>
                  {td.actual !== null && (
                    <div className="text-gray-600 dark:text-gray-300">
                      Actual: <span className="font-mono font-bold">{Math.round(td.actual)} passengers</span>
                    </div>
                  )}
                  {td.predicted !== null && (
                    <div className="text-purple-700 dark:text-purple-300">
                      Forecast: <span className="font-mono font-bold">{Math.round(td.predicted)} passengers</span>
                      {td.confidence_upper !== null && td.confidence_lower !== null && (
                        <span className="text-gray-400 ml-1">
                          (range: {Math.round(td.confidence_lower)}–{Math.round(td.confidence_upper)})
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })()}

            {selectedStation.connected_routes.length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-sm dark:text-gray-300">Connected Routes</h4>
                <div className="flex gap-2 mt-1 flex-wrap">
                  {selectedStation.connected_routes.map((r) => (
                    <button key={r.id} className="px-2 py-1 text-xs rounded-full text-white" style={{ backgroundColor: r.color ?? "#888" }}>{r.name}</button>
                  ))}
                </div>
              </div>
            )}
            {selectedStation.hourly_ridership.length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-sm dark:text-gray-300">Hourly Pattern</h4>
                <div className="flex items-end gap-px h-20 mt-1">
                  {selectedStation.hourly_ridership.map((h) => {
                    const maxR = Math.max(...selectedStation.hourly_ridership.map((x) => x.ridership), 1);
                    const pct = (h.ridership / maxR) * 100;
                    const isRush = (h.hour >= 7 && h.hour <= 9) || (h.hour >= 17 && h.hour <= 19);
                    return (
                      <div key={h.hour} className="flex-1 flex flex-col justify-end">
                        <div className={isRush ? "bg-amber-400" : "bg-gray-300 dark:bg-gray-600"} style={{ height: pct + "%", minHeight: 2 }} />
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[8px] text-gray-400 mt-0.5"><span>0</span><span>6</span><span>12</span><span>18</span><span>23</span></div>
              </div>
            )}
            {selectedStation.forecast.length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-sm dark:text-gray-300">Forecast (next 6 hours)</h4>
                <div className="space-y-1 mt-1">
                  {selectedStation.forecast.slice(0, 6).map((f, i) => (
                    <div key={i} className="flex justify-between text-xs dark:text-gray-300">
                      <span>{new Date(f.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                      <span className="font-mono">{f.predicted} passengers</span>
                      <span className="text-gray-400">{(f.confidence * 100).toFixed(0)}% confidence</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {selectedStation.alerts.length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-sm text-red-600">Active Alerts</h4>
                {selectedStation.alerts.map((a, i) => (
                  <div key={i} className="mt-1 p-2 bg-red-50 dark:bg-red-900/30 rounded text-xs dark:text-red-300">
                    <span className="font-semibold">{a.severity}</span>: {a.title}
                  </div>
                ))}
              </div>
            )}
            {loading && <p className="text-xs text-gray-400 mt-2">Loading...</p>}
          </div>
        )}
      </div>

      {/* Timeline Bar at the bottom */}
      <TimelineBar />
    </div>
  );
}