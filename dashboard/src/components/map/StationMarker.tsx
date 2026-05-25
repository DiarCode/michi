import type { Station } from "@/types";

interface Props { station: Station; onClick?: (stationId: string) => void; highlighted?: boolean }

export default function StationMarker({ station, onClick, highlighted = true }: Props) {
  const r = station.ridership_24h ?? 0;
  const loadPct = station.load_percent ?? (r > 3000 ? 90 : r > 2000 ? 65 : 40);
  const bg = loadPct > 80 ? "bg-red-500" : loadPct > 50 ? "bg-amber-500" : "bg-green-500";
  const opacity = highlighted ? "opacity-100" : "opacity-30";
  const scale = highlighted ? "hover:scale-150" : "";

  return (
    <div
      className={`absolute w-3.5 h-3.5 ${bg} ${opacity} ${scale} rounded-full cursor-pointer border-2 border-white shadow-md transform -translate-x-1/2 -translate-y-1/2 transition-all duration-200`}
      style={{ left: ((station.lon - 71.25) / 0.4) * 100 + "%", top: ((51.25 - station.lat) / 0.3) * 100 + "%" }}
      onClick={() => onClick?.(station.id)}
      title={`${station.name} · ${r} pax/24h · ${loadPct}% load`}
    >
      {highlighted && <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] font-semibold text-gray-800 whitespace-nowrap bg-white/80 px-1 rounded">{station.name}</span>}
    </div>
  );
}
