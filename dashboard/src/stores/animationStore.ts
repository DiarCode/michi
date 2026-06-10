import { create } from "zustand";
import type { BusPosition } from "@/types";

interface InterpolatedBus {
  lat: number;
  lon: number;
  bearing: number;
  speed_kmh: number;
  occupancy_percent: number;
  bus_id: string;
  route_id: string;
  next_stop?: string;
  eta_seconds?: number;
}

interface AnimationStoreState {
  interpolatedPositions: Record<string, InterpolatedBus>;
  /** Trail positions: bus_id → last N interpolated positions */
  trails: Record<string, [number, number][]>;
  /** Whether the animation loop is running */
  running: boolean;
  /** Start the animation loop */
  start: () => void;
  /** Stop the animation loop */
  stop: () => void;
  /** Update target positions from bus store (called on each WS update) */
  updateTargets: (buses: BusPosition[]) => void;
}

const TRAIL_LENGTH = 8;
const INTERPOLATION_FACTOR = 0.15; // How quickly to converge to target position

let rafId: number | null = null;
let currentTargets: Record<string, BusPosition> = {};

function lerpAngle(a: number, b: number, t: number): number {
  let diff = b - a;
  while (diff > Math.PI) diff -= 2 * Math.PI;
  while (diff < -Math.PI) diff += 2 * Math.PI;
  return a + diff * t;
}

function calculateBearing(fromLat: number, fromLon: number, toLat: number, toLon: number): number {
  const dLon = (toLon - fromLon) * (Math.PI / 180);
  const lat1 = fromLat * (Math.PI / 180);
  const lat2 = toLat * (Math.PI / 180);
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return (Math.atan2(y, x) * (180 / Math.PI) + 360) % 360;
}

function animationLoop() {
  const state = useAnimationStore.getState();
  if (!state.running) return;

  const newPositions: Record<string, InterpolatedBus> = {};
  const newTrails: Record<string, [number, number][]> = { ...state.trails };

  for (const [busId, target] of Object.entries(currentTargets)) {
    const current = state.interpolatedPositions[busId];

    if (current) {
      // Interpolate position
      const lat = current.lat + (target.lat - current.lat) * INTERPOLATION_FACTOR;
      const lon = current.lon + (target.lon - current.lon) * INTERPOLATION_FACTOR;
      const bearing = lerpAngle(
        current.bearing * (Math.PI / 180),
        calculateBearing(current.lat, current.lon, target.lat, target.lon) * (Math.PI / 180),
        INTERPOLATION_FACTOR,
      ) * (180 / Math.PI);

      newPositions[busId] = {
        lat,
        lon,
        bearing: ((bearing % 360) + 360) % 360,
        speed_kmh: target.speed_kmh ?? 0,
        occupancy_percent: target.occupancy_percent ?? 0,
        bus_id: target.bus_id,
        route_id: target.route_id,
        next_stop: target.next_stop,
        eta_seconds: target.eta_seconds,
      };
    } else {
      // New bus, start at target position
      const bearing = target.speed_kmh && target.speed_kmh > 0 ? 0 : 0;
      newPositions[busId] = {
        lat: target.lat,
        lon: target.lon,
        bearing,
        speed_kmh: target.speed_kmh ?? 0,
        occupancy_percent: target.occupancy_percent ?? 0,
        bus_id: target.bus_id,
        route_id: target.route_id,
        next_stop: target.next_stop,
        eta_seconds: target.eta_seconds,
      };
    }

    // Update trail
    if (newPositions[busId]) {
      const pos: [number, number] = [newPositions[busId].lon, newPositions[busId].lat];
      const existing = newTrails[busId] ?? [];
      newTrails[busId] = [pos, ...existing.slice(0, TRAIL_LENGTH - 1)];
    }
  }

  // Remove buses that are no longer in targets
  for (const busId of Object.keys(state.interpolatedPositions)) {
    if (!currentTargets[busId]) {
      delete newPositions[busId];
      delete newTrails[busId];
    }
  }

  useAnimationStore.setState({
    interpolatedPositions: newPositions,
    trails: newTrails,
  });

  rafId = requestAnimationFrame(animationLoop);
}

export const useAnimationStore = create<AnimationStoreState>((set) => ({
  interpolatedPositions: {},
  trails: {},
  running: false,

  start: () => {
    const state = useAnimationStore.getState();
    if (state.running) return;
    set({ running: true });
    rafId = requestAnimationFrame(animationLoop);
  },

  stop: () => {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    set({ running: false });
  },

  updateTargets: (buses: BusPosition[]) => {
    const targets: Record<string, BusPosition> = {};
    for (const bus of buses) {
      targets[bus.bus_id] = bus;
    }
    currentTargets = targets;
  },
}));