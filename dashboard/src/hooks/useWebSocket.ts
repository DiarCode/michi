import { useEffect, useState } from "react";
import { wsClient, type WSEvent } from "@/lib/websocket";

export function useWebSocket() {
  const [events, setEvents] = useState<WSEvent[]>([]);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.subscribe((event) => {
      setEvents((prev) => [...prev.slice(-99), event]);
    });
    return () => {
      unsub();
      wsClient.disconnect();
    };
  }, []);

  const busPositions = events
    .filter((e) => e.type === "bus_position")
    .reduce((acc, e) => {
      const id = (e.data as { bus_id: string }).bus_id;
      return { ...acc, [id]: e.data as Record<string, unknown> };
    }, {} as Record<string, Record<string, unknown>>);

  return { events, busPositions };
}
