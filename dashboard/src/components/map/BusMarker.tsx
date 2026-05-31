import type { BusPosition } from "@/types";
import { MapMarker, MarkerContent, MarkerTooltip, MarkerPopup } from "@/components/ui/map";
import { LOAD_HIGH, LOAD_MID } from "@/lib/constants";

interface Props { bus: BusPosition }

export default function BusMarker({ bus }: Props) {
  const occ = bus.occupancy_percent ?? 0;
  const color = occ > LOAD_HIGH ? "#ef4444" : occ > LOAD_MID ? "#f59e0b" : "#22c55e";
  const badgeBg = occ > LOAD_HIGH ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300" : occ > LOAD_MID ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300" : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300";

  return (
    <MapMarker longitude={bus.lon} latitude={bus.lat}>
      <MarkerContent>
        <div className="relative">
          <div
            className="rounded-full border-2 border-white dark:border-gray-900 shadow-lg animate-pulse cursor-pointer"
            style={{ backgroundColor: color, width: 16, height: 16 }}
          />
          <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold whitespace-nowrap bg-white/90 px-1 rounded">
            {bus.bus_id}
          </span>
        </div>
      </MarkerContent>
      <MarkerTooltip>
        <div className="text-xs font-semibold">{bus.bus_id} · Route {bus.route_id}</div>
      </MarkerTooltip>
      <MarkerPopup>
        <div className="p-2 min-w-[160px]">
          <div className="font-bold text-sm">{bus.bus_id}</div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Route: {bus.route_id}</div>
          <div className="text-xs">Speed: {bus.speed_kmh ?? "—"} km/h</div>
          <div className="text-xs">ETA: {bus.eta_seconds ? Math.round(bus.eta_seconds / 60) + " min" : "—"}</div>
          <div className={`inline-block mt-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${badgeBg}`}>
            {occ}% full
          </div>
        </div>
      </MarkerPopup>
    </MapMarker>
  );
}