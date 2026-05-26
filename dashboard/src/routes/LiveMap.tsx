import { useEffect, useState } from "react";
import MapContainer from "@/components/map/MapContainer";
import { useStations } from "@/hooks/useStations";
import { wsClient } from "@/lib/websocket";
import { fetchRoutes, fetchStationDetail, fetchRouteForecast, fetchPredictions } from "@/lib/api";
import type { BusPosition, Route, StationDetail, RouteForecast, PredictionPoint } from "@/types";

const PREDICTION_HORIZONS = [
  { label: "Now", minutes: 0 },
  { label: "+15m", minutes: 15 },
  { label: "+30m", minutes: 30 },
  { label: "+1h", minutes: 60 },
  { label: "+2h", minutes: 120 },
] as const;

export default function LiveMap() {
  const { data } = useStations();
  const stations = data?.stations ?? [];
  const [buses, setBuses] = useState<BusPosition[]>([]);
  const [routes, setRoutes] = useState<Route[]>([]);
  const [selectedRoutes, setSelectedRoutes] = useState<Set<string>>(new Set());
  const [hour, setHour] = useState<number>(new Date().getHours());
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [selectedStation, setSelectedStation] = useState<StationDetail | null>(null);
  const [routeForecast, setRouteForecast] = useState<RouteForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState<PredictionPoint[]>([]);
  const [horizonMinutes, setHorizonMinutes] = useState<number>(0);
  const [currentTime, setCurrentTime] = useState<string>(
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  );

  useEffect(() => { fetchRoutes().then((r) => setRoutes(r.routes ?? [])).catch(() => {}); }, []);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.subscribe((event) => {
      if (event.type === "bus_position") {
        const bus = event.data as unknown as BusPosition;
        setBuses((prev) => [...prev.filter((b) => b.bus_id !== bus.bus_id), bus]);
      }
    });
    return () => { unsub(); wsClient.disconnect(); };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchPredictions(horizonMinutes || undefined)
      .then((r) => setPredictions(r.predictions ?? []))
      .catch(() => setPredictions([]));
  }, [horizonMinutes]);

  const handleStationClick = async (stationId: string) => {
    setLoading(true);
    try { setSelectedStation(await fetchStationDetail(stationId)); }
    catch { setSelectedStation(null); }
    setLoading(false);
  };

  const handleRouteClick = async (routeId: string) => {
    try { setRouteForecast(await fetchRouteForecast(routeId)); }
    catch { setRouteForecast(null); }
  };

  const toggleRoute = (id: string) => {
    setSelectedRoutes((prev) => { const next = new Set(prev); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  };

  const filteredBuses = selectedRoutes.size > 0 ? buses.filter((b) => selectedRoutes.has(b.route_id)) : buses;

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      <div className="w-72 bg-white dark:bg-gray-900 border-r dark:border-gray-700 overflow-y-auto p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-bold text-lg dark:text-white">Live Map</h2>
          <span className="text-xs font-mono text-gray-500 dark:text-gray-400">{currentTime}</span>
        </div>

        <div>
          <h3 className="font-semibold text-sm mb-2 dark:text-gray-300">Prediction Horizon</h3>
          <div className="flex gap-1">
            {PREDICTION_HORIZONS.map((h) => (
              <button
                key={h.minutes}
                onClick={() => setHorizonMinutes(h.minutes)}
                className={`px-2 py-1 text-xs rounded font-medium transition-colors ${
                  horizonMinutes === h.minutes
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                }`}
              >
                {h.label}
              </button>
            ))}
          </div>
        </div>

        <div className="border-t dark:border-gray-700 pt-3">
          <h3 className="font-semibold text-sm mb-2 dark:text-gray-300">Route Filter</h3>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {routes.map((r) => (
              <label key={r.id} className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={selectedRoutes.has(r.id)} onChange={() => toggleRoute(r.id)} className="rounded" />
                <span className="w-3 h-3 rounded-full inline-block flex-shrink-0" style={{ backgroundColor: r.color ?? "#888" }} />
                <span className="text-sm dark:text-gray-300 truncate">{r.name}</span>
                <span className="text-xs text-gray-400 ml-auto">{r.stop_count} stops</span>
              </label>
            ))}
          </div>
        </div>

        <div className="border-t dark:border-gray-700 pt-3">
          <h3 className="font-semibold text-sm mb-2 dark:text-gray-300">Time of Day</h3>
          <input
            type="range"
            min={0}
            max={23}
            value={hour}
            onChange={(e) => setHour(Number(e.target.value))}
            className="w-full accent-blue-600"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">{String(hour).padStart(2, "0")}:00</p>
        </div>

        <div className="border-t dark:border-gray-700 pt-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} />
            <span className="text-sm dark:text-gray-300">Show Heatmap</span>
          </label>
        </div>

        {routeForecast && (
          <div className="border-t dark:border-gray-700 pt-3">
            <h3 className="font-semibold text-sm mb-2 dark:text-gray-300">{routeForecast.route?.name ?? routeForecast.route_id}</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">{routeForecast.stop_count} stops · avg {routeForecast.avg_ridership}/hr</p>
            <div className="mt-2 space-y-1 max-h-40 overflow-y-auto">
              {routeForecast.forecast.slice(0, 8).map((f) => (
                <div key={f.hour} className="flex justify-between text-xs dark:text-gray-400">
                  <span>{String(f.hour).padStart(2, "0")}:00</span>
                  <span className="font-mono">{f.predicted} pax</span>
                  <span className="text-gray-400">{(f.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {predictions.length > 0 && (
          <div className="border-t dark:border-gray-700 pt-3">
            <h3 className="font-semibold text-sm mb-1 dark:text-gray-300">
              Predictions {horizonMinutes > 0 ? `(+${horizonMinutes}m)` : "(Now)"}
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">{predictions.length} stations</p>
          </div>
        )}
      </div>

      <div className="flex-1 relative">
        <MapContainer
          stations={stations}
          buses={filteredBuses}
          hour={hour}
          onStationClick={handleStationClick}
          showHeatmap={showHeatmap}
          predictions={predictions}
        />

        {selectedStation && (
          <div className="absolute top-0 right-0 w-80 h-full bg-white/95 dark:bg-gray-900/95 shadow-lg overflow-y-auto p-4">
            <button onClick={() => setSelectedStation(null)} className="absolute top-2 right-2 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 text-lg">✕</button>
            <h3 className="font-bold text-lg dark:text-white">{selectedStation.station.name}</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">{selectedStation.station.district} · {selectedStation.station.ridership_24h} pax/24h</p>
            {selectedStation.connected_routes.length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-sm dark:text-gray-300">Connected Routes</h4>
                <div className="flex gap-2 mt-1 flex-wrap">
                  {selectedStation.connected_routes.map((r) => (
                    <button key={r.id} onClick={() => handleRouteClick(r.id)} className="px-2 py-1 text-xs rounded-full text-white" style={{ backgroundColor: r.color ?? "#888" }}>{r.name}</button>
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
                    const isNow = h.hour === hour;
                    const isRush = (h.hour >= 7 && h.hour <= 9) || (h.hour >= 17 && h.hour <= 19);
                    return (
                      <div key={h.hour} className="flex-1 flex flex-col justify-end">
                        <div className={isNow ? "bg-blue-600" : isRush ? "bg-amber-400" : "bg-gray-300 dark:bg-gray-600"} style={{ height: pct + "%", minHeight: 2 }} />
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[8px] text-gray-400 mt-0.5"><span>0</span><span>6</span><span>12</span><span>18</span><span>23</span></div>
              </div>
            )}
            {selectedStation.forecast.length > 0 && (
              <div className="mt-3">
                <h4 className="font-semibold text-sm dark:text-gray-300">Forecast (next 6h)</h4>
                <div className="space-y-1 mt-1">
                  {selectedStation.forecast.slice(0, 6).map((f, i) => (
                    <div key={i} className="flex justify-between text-xs dark:text-gray-300">
                      <span>{new Date(f.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                      <span className="font-mono">{f.predicted} pax</span>
                      <span className="text-gray-400">{(f.confidence * 100).toFixed(0)}%</span>
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
    </div>
  );
}