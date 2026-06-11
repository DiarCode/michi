import { create } from "zustand"
import type { ConnectionStatus } from "@/types"
import { wsClient, type WSEvent } from "@/lib/websocket"

interface ConnectionStoreState extends ConnectionStatus {
  init: () => () => void
}

export const useConnectionStore = create<ConnectionStoreState>((set, get) => ({
  connected: false,
  lastTickReceived: 0,
  reconnectAttempt: 0,
  lastConnectedAt: null,

  // Subscribe to WS events to track connection status.
  // Call from a top-level component once.
  init: () => {
    wsClient.connect()
    const unsub = wsClient.subscribe((event: WSEvent) => {
      if (
        event.type === "simulation_tick" &&
        typeof event.data?.tick === "number"
      ) {
        set({
          lastTickReceived: event.data.tick as number,
          connected: true,
          lastConnectedAt: new Date().toISOString(),
        })
      }
      // Any message means connected
      set({ connected: true, lastConnectedAt: new Date().toISOString() })
    })

    // Periodic stale detection: if no tick received in 30s, mark disconnected
    const staleInterval = setInterval(() => {
      const state = get()
      if (state.lastConnectedAt) {
        const elapsed = Date.now() - new Date(state.lastConnectedAt).getTime()
        if (elapsed > 30000) {
          set({ connected: false })
        }
      }
    }, 10000)

    return () => {
      unsub()
      clearInterval(staleInterval)
    }
  },
}))
