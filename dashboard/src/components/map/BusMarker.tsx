import type { BusPosition } from "@/types";
import { MAP_LON_MIN, MAP_LON_SPAN, MAP_LAT_MAX, MAP_LAT_SPAN, LOAD_HIGH, LOAD_MID } from "@/lib/constants";

interface Props { bus: BusPosition }

export default function BusMarker({ bus }: Props) {
  const occ = bus.occupancy_percent ?? 0;
  const bg = occ > LOAD_HIGH ? "bg-red-500" : occ > LOAD_MID ? "bg-amber-500" : "bg-green-500";
  const badgeBg = occ > LOAD_HIGH ? "bg-red-100 text-red-700" : occ > LOAD_MID ? "bg-amber-100 text-amber-700" : "bg-green-100 text-green-700";

  return (
    <div className="absolute group" style={{ left: ((bus.lon - MAP_LON_MIN) / MAP_LON_SPAN) * 100 + "%", top: ((MAP_LAT_MAX - bus.lat) / MAP_LAT_SPAN) * 100 + "%", transform: "translate(-50%, -50%)" }}>
      <div className={`w-4 h-4 ${bg} rounded-full border-2 border-white shadow-lg animate-pulse cursor-pointer`}>
        <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold whitespace-nowrap">{bus.bus_id}</span>
      </div>
      <div className="hidden group-hover:block absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-white rounded-lg shadow-lg p-2 text-xs z-50 min-w-[140px]">
        <div className="font-bold mb-1">{bus.bus_id}</div>
        <div>Route: {bus.route_id}</div>
        <div>Speed: {bus.speed_kmh ?? "—"} km/h</div>
        <div>ETA: {bus.eta_seconds ? Math.round(bus.eta_seconds / 60) + " min" : "—"}</div>
        <div className={`inline-block mt-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${badgeBg}`}>{occ}% full</div>
      </div>
    </div>
  );
}