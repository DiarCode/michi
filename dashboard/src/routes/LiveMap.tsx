import { useEffect, useState } from "react";
import MapContainer from "@/components/map/MapContainer";
import { useStations } from "@/hooks/useStations";
import { wsClient } from "@/lib/websocket";
import type { BusPosition } from "@/types";

export default function LiveMap() {
  const { data } = useStations();
  const stations = data?.stations ?? [];
  const [buses, setBuses] = useState<BusPosition[]>([]);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.subscribe((event) => {
      if (event.type === "bus_position") {
        const bus = event.data as unknown as BusPosition;
        setBuses((prev) => {
          const filtered = prev.filter((b) => b.bus_id !== bus.bus_id);
          return [...filtered, bus];
        });
      }
    });
    return () => { unsub(); wsClient.disconnect(); };
  }, []);

  return (
    <div className="h-[calc(100vh-4rem)]">
      <MapContainer stations={stations} buses={buses} />
    </div>
  );
}
