import type { BusPosition } from "@/types";

interface Props { bus: BusPosition }

export default function BusMarker({ bus }: Props) {
  return (
    <div className="absolute w-4 h-4 bg-green-500 rounded-full border-2 border-white shadow-lg animate-pulse transform -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${((bus.lon - 71.25) / 0.4) * 100}%`, top: `${((51.25 - bus.lat) / 0.3) * 100}%` }}
      title={`${bus.bus_id} · ${bus.speed_kmh} km/h · ${bus.occupancy_percent}% full`}
    >
      <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold text-green-700 whitespace-nowrap">{bus.bus_id}</span>
    </div>
  );
}
