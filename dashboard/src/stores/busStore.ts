import { create } from "zustand";
import type { BusPosition } from "@/types";
import { wsClient, type WSEvent } from "@/lib/websocket";
import { useQueryClient } from "@tanstack/react-query";

interface BusState {
  buses: Record<string, BusPosition>;
  subscribe: () => () => void;
}

// We need a reference to the queryClient for invalidation.
// Zustand stores run outside React, so we access it via a module-level reference.
let _qc: ReturnType<typeof useQueryClient> | null = null;

/** Call once from a React component to wire up TanStack Query invalidation. */
export function initBusStoreInvalidation(qc: ReturnType<typeof useQueryClient>) {
  _qc = qc;
}

export const useBusStore = create<BusState>((set) => {
  let unsub: (() => void) | null = null;

  return {
    buses: {},

    subscribe: () => {
      wsClient.connect();
      unsub = wsClient.subscribe((event: WSEvent) => {
        if (event.type === "bus_position") {
          const bus = event.data as unknown as BusPosition;
          set((state) => ({
            buses: { ...state.buses, [bus.bus_id]: bus },
          }));
          // Invalidate relevant TanStack Query keys when WS updates server state
          if (_qc) {
            _qc.invalidateQueries({ queryKey: ["stations"] });
          }
        }
      });
      return () => {
        if (unsub) {
          unsub();
          unsub = null;
        }
      };
    },
  };
});