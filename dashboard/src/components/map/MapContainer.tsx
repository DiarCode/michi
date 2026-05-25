import type { ReactNode } from "react";
import type { Station, BusPosition } from "@/types";

interface Props {
  stations: Station[];
  buses: BusPosition[];
  children?: ReactNode;
}

export default function MapContainer({ stations, buses, children }: Props) {
  const center = stations.length > 0 ? `${stations[0].lat},${stations[0].lon}` : "51.1605,71.4704";
  const markersParam = stations.map((s) => `${s.lat},${s.lon},${encodeURIComponent(s.name)}`).join("|");

  return (
    <div className="relative w-full h-full">
      <iframe
        src={`https://mapcn.dev/embed?center=${center}&zoom=13&markers=${markersParam}`}
        className="w-full h-full border-0"
        allow="geolocation"
        title="Astana Transit Map"
      />
      <div className="absolute top-4 left-4 bg-white/90 p-3 rounded-lg shadow-md">
        <h3 className="font-bold text-sm">Live Tracking</h3>
        <p className="text-xs text-gray-600">{buses.length} buses active</p>
        {children}
      </div>
    </div>
  );
}
