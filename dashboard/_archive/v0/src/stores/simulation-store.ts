import { create } from "zustand"

interface SimulationState {
  running: boolean
  tick: number
  drift: "stable" | "drifting" | "critical"
  lastUpdate: string | null
  setState: (next: Partial<SimulationState>) => void
}

export const useSimulationStore = create<SimulationState>((set) => ({
  running: false,
  tick: 0,
  drift: "stable",
  lastUpdate: null,
  setState: (next) => set(next),
}))
