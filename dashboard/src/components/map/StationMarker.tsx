import type { Station } from "@/types";
import { MAP_LON_MIN, MAP_LON_SPAN, MAP_LAT_MAX, MAP_LAT_SPAN, LOAD_HIGH, LOAD_MID, RIDERSHIP_HIGH, RIDERSHIP_MID } from "@/lib/constants";

interface Props { station: Station; onClick?: (stationId: string) => void; highlighted?: boolean }

export default function StationMarker({ station, onClick, highlighted = true }: Props) {
  const r = station.ridership_24h ?? 0;
  const loadPct = station.load_percent ?? (r > RIDERSHIP_HIGH ? 90 : r > RIDERSHIP_MID ? 65 : 40);
  const bg = loadPct > LOAD_HIGH ? "bg-red-500" : loadPct > LOAD_MID ? "bg-amber-500" : "bg-green-500";
  const opacity = highlighted ? "opacity-100" : "opacity-30";
  const scale = highlighted ? "hover:scale-150" : "";

  return (
    <div
      className={`absolute w-3.5 h-3.5 ${bg} ${opacity} ${scale} rounded-full cursor-pointer border-2 border-white shadow-md transform -translate-x-1/2 -translate-y-1/2 transition-all duration-200`}
      style={{ left: ((station.lon - MAP_LON_MIN) / MAP_LON_SPAN) * 100 + "%", top: ((MAP_LAT_MAX - station.lat) / MAP_LAT_SPAN) * 100 + "%" }}
      onClick={() => onClick?.(station.id)}
      title={`${station.name} · ${r} pax/24h · ${loadPct}% load`}
    >
      {highlighted && <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] font-semibold text-gray-800 whitespace-nowrap bg-white/80 px-1 rounded">{station.name}</span>}
    </div>
  );
}