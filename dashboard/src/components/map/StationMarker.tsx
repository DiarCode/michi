import type { Station } from "@/types";
import { MapMarker, MarkerContent, MarkerTooltip, MarkerPopup } from "@/components/ui/map";
import { LOAD_HIGH, LOAD_MID, STATION_CAPACITY, MORNING_PEAK, EVENING_PEAK } from "@/lib/constants";

interface Props {
  station: Station;
  onClick?: (stationId: string) => void;
  hour?: number;
  predictedLoad?: number;
}

function getLoadPercent(station: Station, hour: number): number {
  const base = station.ridership_24h ?? 1000;
  if (hour >= MORNING_PEAK[0] && hour <= MORNING_PEAK[1]) return Math.min(95, Math.round(base / STATION_CAPACITY * 100));
  if (hour >= EVENING_PEAK[0] && hour <= EVENING_PEAK[1]) return Math.min(95, Math.round(base / STATION_CAPACITY * 100));
  if (hour >= 6 && hour <= 22) return Math.min(70, Math.round(base * 0.6 / STATION_CAPACITY * 100));
  return Math.min(30, Math.round(base * 0.15 / STATION_CAPACITY * 100));
}

export default function StationMarker({ station, onClick, hour = new Date().getHours(), predictedLoad }: Props) {
  const load = getLoadPercent(station, hour);
  const color = load > LOAD_HIGH ? "#ef4444" : load > LOAD_MID ? "#f59e0b" : "#22c55e";
  const size = load > LOAD_HIGH ? 14 : load > LOAD_MID ? 11 : 8;

  return (
    <MapMarker longitude={station.lon} latitude={station.lat}>
      <MarkerContent>
        <div
          className="rounded-full border-2 border-white shadow-md cursor-pointer transition-transform hover:scale-150"
          style={{ backgroundColor: color, width: size, height: size }}
          onClick={() => onClick?.(station.id)}
        />
      </MarkerContent>
      <MarkerTooltip>
        <div className="text-xs">
          <div className="font-semibold">{station.name}</div>
          <div className="text-gray-500">{station.ridership_24h ?? "—"} pax/24h · {load}% load</div>
        </div>
      </MarkerTooltip>
      <MarkerPopup>
        <div className="p-2 min-w-[180px]">
          <div className="font-bold text-sm">{station.name}</div>
          {station.district && <div className="text-xs text-gray-500">{station.district}</div>}
          <div className="mt-1 text-xs">
            <span>Ridership: </span>
            <span className="font-mono">{station.ridership_24h ?? "—"}</span>
            <span> pax/24h</span>
          </div>
          <div className="text-xs">
            <span>Load: </span>
            <span className="font-mono font-bold" style={{ color }}>{load}%</span>
          </div>
          {predictedLoad !== undefined && (
            <div className="text-xs mt-1 pt-1 border-t border-gray-200">
              <span>Predicted: </span>
              <span className="font-mono font-bold text-blue-600">{predictedLoad} pax</span>
            </div>
          )}
        </div>
      </MarkerPopup>
    </MapMarker>
  );
}