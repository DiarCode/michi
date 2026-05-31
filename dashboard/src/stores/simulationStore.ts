import { create } from "zustand";
import type { SimulationState, SimulationTick, ValidationMetric, DriftAlert } from "@/types";
import { wsClient, type WSEvent } from "@/lib/websocket";

interface SimulationStoreState extends SimulationState {
  /** Handle a simulation_tick WS event */
  handleTick: (data: Record<string, unknown>) => void;
  /** Handle a validation_metric WS event */
  handleValidationMetric: (data: Record<string, unknown>) => void;
  /** Handle a drift_alert WS event */
  handleDriftAlert: (data: Record<string, unknown>) => void;
  /** Handle a combined SimulationTick (legacy path) */
  updateFromTick: (tick: SimulationTick) => void;
  /** Mark data as stale */
  markStale: () => void;
  /** Subscribe to WS simulation events. Call once from a React component. */
  subscribe: () => () => void;
  /** Start the simulation */
  startSimulation: () => void;
  /** Stop the simulation */
  stopSimulation: () => void;
}

export const useSimulationStore = create<SimulationStoreState>((set, get) => ({
  running: false,
  tick: 0,
  startTime: null,
  metricsHistory: [],
  driftAlerts: [],
  isStale: false,
  lastTickAt: null,

  startSimulation: () =>
    set({
      running: true,
      tick: 0,
      startTime: new Date().toISOString(),
      metricsHistory: [],
      driftAlerts: [],
      isStale: false,
      lastTickAt: new Date().toISOString(),
    }),

  stopSimulation: () => set({ running: false }),

  updateFromTick: (simTick: SimulationTick) =>
    set((state) => ({
      tick: simTick.tick,
      lastTickAt: new Date().toISOString(),
      isStale: false,
      running: true,
      metricsHistory: [...state.metricsHistory.slice(-299), simTick.metrics],
      driftAlerts: simTick.events
        ? simTick.events
            .filter((e) => e.type === "drift_alert")
            .map(
              (e): DriftAlert => ({
                metric: e.type,
                current_value: 0,
                baseline_value: 0,
                deviation_pct: 0,
                severity: "medium",
                timestamp: simTick.timestamp,
              })
            )
        : state.driftAlerts,
    })),

  /** Handle simulation_tick WS event data */
  handleTick: (data) => {
    const tickValue = data.tick as number | undefined;
    set((state) => ({
      tick: typeof tickValue === "number" ? tickValue : state.tick,
      lastTickAt: new Date().toISOString(),
      running: true,
      isStale: false,
    }));
  },

  /** Handle validation_metric WS event data */
  handleValidationMetric: (data) => {
    const metric: ValidationMetric = {
      mae: (data.mae as number) ?? 0,
      mape: (data.mape as number) ?? 0,
      accuracy: data.accuracy as number | undefined,
      drift_status: data.drift_status as ValidationMetric["drift_status"],
      tick: data.tick as number | undefined,
      timestamp: data.timestamp as string | undefined,
    };
    set((state) => ({
      metricsHistory: [...state.metricsHistory.slice(-299), metric],
      isStale: false,
      lastTickAt: new Date().toISOString(),
    }));
  },

  /** Handle drift_alert WS event data */
  handleDriftAlert: (data) => {
    const alert: DriftAlert = {
      metric: (data.metric as string) ?? "unknown",
      current_value: (data.current_value as number) ?? 0,
      baseline_value: (data.baseline_value as number) ?? 0,
      deviation_pct: (data.deviation_pct as number) ?? 0,
      severity: (data.severity as DriftAlert["severity"]) ?? "medium",
      timestamp: (data.timestamp as string) ?? new Date().toISOString(),
    };
    set((state) => ({
      driftAlerts: [...state.driftAlerts.slice(-49), alert],
    }));
  },

  markStale: () => set({ isStale: true }),

  /** Subscribe to WS simulation-related events */
  subscribe: () => {
    wsClient.connect();
    const unsub = wsClient.subscribe((event: WSEvent) => {
      const { handleTick, handleValidationMetric, handleDriftAlert } = get();
      switch (event.type) {
        case "simulation_tick":
          handleTick(event.data);
          break;
        case "validation_metric":
          handleValidationMetric(event.data);
          break;
        case "drift_alert":
          handleDriftAlert(event.data);
          break;
      }
    });
    return unsub;
  },
}));