import type { ReactNode } from "react";
import type { Station, BusPosition } from "@/types";
import StationMarker from "./StationMarker";
import BusMarker from "./BusMarker";
import {
  MAP_LON_MIN, MAP_LON_SPAN, MAP_LAT_MAX, MAP_LAT_SPAN,
  LOAD_HIGH, LOAD_MID, STATION_CAPACITY,
  HEATMAP_MIN_PX, HEATMAP_RANGE_PX,
  MORNING_PEAK, EVENING_PEAK,
} from "@/lib/constants";

interface Props {
  stations: Station[];
  buses: BusPosition[];
  showHeatmap?: boolean;
  hour?: number;
  onStationClick?: (stationId: string) => void;
  selectedRoutes?: Set<string>;
  children?: ReactNode;
}

const STATION_ROUTES: Record<string, string[]> = {
  S001: ["R1", "R4"], S002: ["R2"], S003: ["R1", "R2", "R5"],
  S004: ["R3"], S005: ["R3"], S006: ["R3"],
  S007: ["R2"], S008: ["R4"], S009: ["R4"],
  S010: ["R1", "R5"], S011: ["R3"], S012: ["R5"],
};

export default function MapContainer({ stations, buses, showHeatmap = true, hour = new Date().getHours(), onStationClick, selectedRoutes }: Props) {
  const getHeatColor = (load: number) => load > LOAD_HIGH ? "#ef4444" : load > LOAD_MID ? "#f59e0b" : "#22c55e";
  const getLoadPercent = (s: Station) => {
    const base = s.ridership_24h ?? 1000;
    if (hour >= MORNING_PEAK[0] && hour <= MORNING_PEAK[1] || hour >= EVENING_PEAK[0] && hour <= EVENING_PEAK[1]) return Math.min(95, Math.round(base / STATION_CAPACITY * 100));
    if (hour >= 6 && hour <= 22) return Math.min(70, Math.round(base * 0.6 / STATION_CAPACITY * 100));
    return Math.min(30, Math.round(base * 0.15 / STATION_CAPACITY * 100));
  };
  const isHighlighted = (s: Station) => {
    if (!selectedRoutes || selectedRoutes.size === 0) return true;
    return (STATION_ROUTES[s.id] ?? []).some((r) => selectedRoutes.has(r));
  };

  return (
    <div className="relative w-full h-full bg-gray-100">
      {/* Grid background */}
      <div className="absolute inset-0" style={{ backgroundImage: "linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)", backgroundSize: "20px 20px" }} />

      {/* Heatmap circles */}
      {showHeatmap && stations.map((s) => {
        const load = getLoadPercent(s);
        const size = HEATMAP_MIN_PX + (load / 100) * HEATMAP_RANGE_PX;
        const left = ((s.lon - MAP_LON_MIN) / MAP_LON_SPAN) * 100;
        const top = ((MAP_LAT_MAX - s.lat) / MAP_LAT_SPAN) * 100;
        const hl = isHighlighted(s);
        return (<div key={"heat-" + s.id} className="absolute rounded-full transition-all duration-300" style={{ left: left + "%", top: top + "%", width: size, height: size, backgroundColor: getHeatColor(load), transform: "translate(-50%, -50%)", opacity: hl ? 0.6 : 0.15 }} />);
      })}

      {/* Station markers */}
      {stations.map((s) => (<StationMarker key={s.id} station={s} onClick={onStationClick} highlighted={isHighlighted(s)} />))}

      {/* Bus markers */}
      {buses.map((b) => (<BusMarker key={b.bus_id} bus={b} />))}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/90 p-2 rounded shadow text-xs space-y-1">
        <div className="font-semibold">Load Level</div>
        <div className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500" /> &lt;50%</div>
        <div className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-amber-500" /> 50-80%</div>
        <div className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500" /> &gt;80%</div>
      </div>

      {/* Info overlay */}
      <div className="absolute top-4 left-4 bg-white/90 p-3 rounded-lg shadow-md">
        <h3 className="font-bold text-sm">Live Tracking</h3>
        <p className="text-xs text-gray-600">{buses.length} buses active · {String(hour).padStart(2, "0")}:00</p>
      </div>
    </div>
  );
}