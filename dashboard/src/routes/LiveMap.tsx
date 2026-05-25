import { useEffect, useState } from "react";
import type { BusPosition } from "../types";

export default function LiveMap() {
  const [buses, setBuses] = useState<BusPosition[]>([]);
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/realtime");
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "bus_position") {
        setBuses((prev) => [...prev.filter((b) => b.bus_id !== msg.data.bus_id), msg.data]);
      }
    };
    return () => ws.close();
  }, []);
  return (
    <div className="h-[calc(100vh-4rem)] relative">
      <iframe
        src="https://mapcn.dev/embed?center=51.1605,71.4702&zoom=13"
        className="w-full h-full border-0"
        title="Astana Map"
      />
      <div className="absolute top-4 left-4 bg-white/90 p-3 rounded shadow">
        <h3 className="font-bold text-sm">Active Buses</h3>
        <p className="text-xs text-gray-600">{buses.length} buses tracked</p>
      </div>
    </div>
  );
}
