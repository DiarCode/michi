import { useEffect, useState } from "react";
import { wsClient, type WSEvent } from "@/lib/websocket";
import { useSimulationStore } from "@/stores/simulationStore";

export function useWebSocket() {
  const [events, setEvents] = useState<WSEvent[]>([]);

  const subscribeSim = useSimulationStore((s) => s.subscribe);

  useEffect(() => {
    wsClient.connect();

    // Subscribe to all events for local state
    const unsub = wsClient.subscribe((event) => {
      setEvents((prev) => [...prev.slice(-99), event]);
    });

    // Subscribe simulation store to simulation-related events
    const unsubSim = subscribeSim();

    // Wire up TanStack Query invalidation for alert events
    // Access queryClient via the module-level reference in busStore
    // For alert invalidation, we use a direct subscribe listener

    const unsubAlerts = wsClient.subscribe((event) => {
      // Invalidate TanStack Query keys for relevant data changes
      if (event.type === "alert" || event.type === "forecast_update") {
        // We can't use hooks here, but busStore already handles station invalidation.
        // Alert and forecast events will be handled by polling/refetch intervals.
      }
    });

    return () => {
      unsub();
      unsubSim();
      unsubAlerts();
      wsClient.disconnect();
    };
  }, [subscribeSim]);

  const busPositions = events
    .filter((e) => e.type === "bus_position")
    .reduce((acc, e) => {
      const id = (e.data as { bus_id: string }).bus_id;
      return { ...acc, [id]: e.data as Record<string, unknown> };
    }, {} as Record<string, Record<string, unknown>>);

  return { events, busPositions };
}