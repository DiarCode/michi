import type { Station } from "@/types";

interface Props { station: Station; onClick?: (s: Station) => void }

export default function StationMarker({ station, onClick }: Props) {
  const r = station.ridership_24h ?? 0;
  const bg = r > 3000 ? "bg-red-500" : r > 2000 ? "bg-amber-500" : "bg-blue-500";
  return (
    <div className={`absolute w-3 h-3 ${bg} rounded-full cursor-pointer border-2 border-white shadow-md transform -translate-x-1/2 -translate-y-1/2 hover:scale-150 transition-transform`}
      style={{ left: `${((station.lon - 71.25) / 0.4) * 100}%`, top: `${((51.25 - station.lat) / 0.3) * 100}%` }}
      onClick={() => onClick?.(station)}
      title={station.name}
    />
  );
}
