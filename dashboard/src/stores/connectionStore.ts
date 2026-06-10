import { create } from "zustand";
import type { ConnectionStatus } from "@/types";
import { wsClient, type WSEvent, type WSConnectionState } from "@/lib/websocket";

interface ConnectionStoreState extends ConnectionStatus {
  wsState: WSConnectionState;
  init: () => () => void;
}

export const useConnectionStore = create<ConnectionStoreState>((set, get) => ({
  connected: false,
  lastTickReceived: 0,
  reconnectAttempt: 0,
  lastConnectedAt: null,
  wsState: "disconnected",

  // Subscribe to WS events and state changes.
  // Call from a top-level component once.
  init: () => {
    wsClient.connect();

    // Listen for connection state changes from WSClient
    const unsubState = wsClient.onStateChange((state: WSConnectionState) => {
      set({
        wsState: state,
        connected: state === "connected",
        reconnectAttempt: state === "connecting" ? get().reconnectAttempt + 1 : 0,
        lastConnectedAt: state === "connected" ? new Date().toISOString() : get().lastConnectedAt,
      });
    });

    // Listen for events to track simulation ticks
    const unsub = wsClient.subscribe((event: WSEvent) => {
      if (event.type === "simulation_tick" && typeof event.data?.tick === "number") {
        set({
          lastTickReceived: event.data.tick as number,
          connected: true,
          lastConnectedAt: new Date().toISOString(),
        });
      }
      // Any message means connected
      set({ connected: true, lastConnectedAt: new Date().toISOString() });
    });

    // Periodic stale detection: if no tick received in 30s, mark disconnected
    const staleInterval = setInterval(() => {
      const state = get();
      if (state.lastConnectedAt) {
        const elapsed = Date.now() - new Date(state.lastConnectedAt).getTime();
        if (elapsed > 30000) {
          set({ connected: false });
        }
      }
    }, 10000);

    return () => {
      unsubState();
      unsub();
      clearInterval(staleInterval);
    };
  },
}));