import type { BusPosition } from "@/types";
import { MapMarker, MarkerContent, MarkerTooltip, MarkerPopup } from "@/components/ui/map";
import { LOAD_HIGH, LOAD_MID } from "@/lib/constants";

interface Props { bus: BusPosition; routeColor?: string }

export default function BusMarker({ bus, routeColor }: Props) {
  const occ = bus.occupancy_percent ?? 0;
  const loadColor = occ > LOAD_HIGH ? "#ef4444" : occ > LOAD_MID ? "#f59e0b" : "#22c55e";
  // Use route color as the border, occupancy as the fill — gives clear route identity
  const markerColor = routeColor ?? loadColor;
  const borderColor = routeColor ? loadColor : "#ffffff";
  const badgeBg = occ > LOAD_HIGH ? "bg-michi-red/10 text-michi-red" : occ > LOAD_MID ? "bg-michi-amber/10 text-michi-amber" : "bg-michi-lime/15 text-michi-lime-dark";

  return (
    <MapMarker longitude={bus.lon} latitude={bus.lat}>
      <MarkerContent>
        <div className="relative">
          <div
            className="rounded-full border-2 shadow-lg animate-pulse cursor-pointer"
            style={{ backgroundColor: markerColor, borderColor, width: 18, height: 18 }}
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
          <div className="text-xs text-michi-muted mt-1">Route: {bus.route_id}</div>
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