import { create } from "zustand"

interface ConnectionState {
  online: boolean
  lastPing: number | null
  setOnline: (online: boolean) => void
  setLastPing: (ping: number | null) => void
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  online: typeof navigator === "undefined" ? true : navigator.onLine,
  lastPing: null,
  setOnline: (online) => set({ online }),
  setLastPing: (lastPing) => set({ lastPing }),
}))

export function setupConnectionListeners() {
  if (typeof window === "undefined") return () => {}
  const onOnline = () => useConnectionStore.getState().setOnline(true)
  const onOffline = () => useConnectionStore.getState().setOnline(false)
  window.addEventListener("online", onOnline)
  window.addEventListener("offline", onOffline)
  return () => {
    window.removeEventListener("online", onOnline)
    window.removeEventListener("offline", onOffline)
  }
}
