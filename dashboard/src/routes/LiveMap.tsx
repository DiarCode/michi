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

  const visibleBuses = timelineMode === "historical" ? [] : buses;

  if (isLoading) {
    return <div className="h-[calc(100vh-4rem)] bg-michi-page animate-pulse" />;
  }

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
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

        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="absolute top-4 right-4 z-10 px-4 py-2 text-sm font-semibold rounded-full shadow-card transition-all bg-white/95 backdrop-blur-sm text-michi-body hover:bg-white border border-michi-border"
        >
          {showHeatmap ? "● Heatmap" : "○ Heatmap"}
        </button>

        {visibleBuses.length > 0 && (
          <div className="absolute top-4 left-4 z-10 px-4 py-2 text-sm font-semibold rounded-full shadow-card bg-michi-dark/90 backdrop-blur-sm text-white">
            <span className="inline-block w-2 h-2 rounded-full bg-michi-lime mr-2" />
            {visibleBuses.length} buses · {String(timelineHour).padStart(2, "0")}:00
          </div>
        )}

        {selectedStation && (
          <div className="absolute top-0 right-0 w-80 h-full bg-white/95 shadow-lg overflow-y-auto p-5 border-l border-michi-border">
            <button onClick={() => setSelectedStation(null)} className="absolute top-3 right-3 w-7 h-7 rounded-full bg-michi-warm flex items-center justify-center text-michi-muted hover:text-michi-dark hover:bg-michi-border transition-colors text-sm">✕</button>
            <h3 className="font-bold text-lg text-michi-dark">{selectedStation.station.name}</h3>
            <p className="text-sm text-michi-muted mt-0.5">{selectedStation.station.district} · {selectedStation.station.ridership_24h?.toLocaleString() ?? "—"} passengers/day</p>

            {timelineMode === "historical" && (() => {
              const td = getTimelineStationData(selectedStation.station.id);
              if (!td) return null;
              return (
                <div className="mt-3 p-3 bg-michi-lime/10 rounded-xl text-sm">
                  <div className="font-semibold text-michi-lime-dark mb-1.5">Timeline Data</div>
                  {td.actual !== null && (
                    <div className="text-michi-body">
                      Actual: <span className="font-mono font-bold">{Math.round(td.actual)} passengers</span>
                    </div>
                  )}
                  {td.predicted !== null && (
                    <div className="text-michi-teal">
                      Forecast: <span className="font-mono font-bold">{Math.round(td.predicted)} passengers</span>
                      {td.confidence_upper !== null && td.confidence_lower !== null && (
                        <span className="text-michi-muted ml-1">
                          (range: {Math.round(td.confidence_lower)}–{Math.round(td.confidence_upper)})
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })()}

            {selectedStation.connected_routes.length > 0 && (
              <div className="mt-4">
                <h4 className="font-semibold text-sm text-michi-dark mb-2">Connected Routes</h4>
                <div className="flex gap-2 flex-wrap">
                  {selectedStation.connected_routes.map((r) => (
                    <button key={r.id} className="px-3 py-1 text-xs rounded-full text-white font-semibold" style={{ backgroundColor: r.color ?? "#888" }}>{r.name}</button>
                  ))}
                </div>
              </div>
            )}
            {selectedStation.hourly_ridership.length > 0 && (
              <div className="mt-4">
                <h4 className="font-semibold text-sm text-michi-dark mb-2">Hourly Pattern</h4>
                <div className="flex items-end gap-px h-20 mt-1">
                  {selectedStation.hourly_ridership.map((h) => {
                    const maxR = Math.max(...selectedStation.hourly_ridership.map((x) => x.ridership), 1);
                    const pct = (h.ridership / maxR) * 100;
                    const isRush = (h.hour >= 7 && h.hour <= 9) || (h.hour >= 17 && h.hour <= 19);
                    return (
                      <div key={h.hour} className="flex-1 flex flex-col justify-end">
                        <div className={isRush ? "bg-michi-lime" : "bg-michi-border"} style={{ height: pct + "%", minHeight: 2, borderRadius: '2px' }} />
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[9px] text-michi-muted mt-1"><span>0</span><span>6</span><span>12</span><span>18</span><span>23</span></div>
              </div>
            )}
            {selectedStation.forecast.length > 0 && (
              <div className="mt-4">
                <h4 className="font-semibold text-sm text-michi-dark mb-2">Forecast (next 6 hours)</h4>
                <div className="space-y-1.5 mt-1">
                  {selectedStation.forecast.slice(0, 6).map((f, i) => (
                    <div key={i} className="flex justify-between text-sm text-michi-body">
                      <span>{new Date(f.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                      <span className="font-mono font-semibold">{f.predicted} pax</span>
                      <span className="text-michi-muted">{(f.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {selectedStation.alerts.length > 0 && (
              <div className="mt-4">
                <h4 className="font-semibold text-sm text-michi-red mb-2">Active Alerts</h4>
                {selectedStation.alerts.map((a, i) => (
                  <div key={i} className="mt-1.5 p-2.5 bg-michi-red/8 border-l-4 border-l-michi-red rounded-xl text-sm text-michi-red">
                    <span className="font-semibold uppercase text-xs">{a.severity}</span>: {a.title}
                  </div>
                ))}
              </div>
            )}
            {loading && <p className="text-sm text-michi-muted mt-3">Loading...</p>}
          </div>
        )}
      </div>

      <TimelineBar />
    </div>
  );
}