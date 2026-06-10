import { create } from "zustand";
import type { TimelineMode, TimelinePoint } from "@/types";

interface TimelineState {
  currentTime: number;
  isPlaying: boolean;
  playSpeed: number;
  range: { start: number; end: number };
  data: TimelinePoint[];
  mode: TimelineMode;

  scrubTo: (timestamp: number) => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  setSpeed: (speed: number) => void;
  enterLiveMode: () => void;
  enterHistoricalMode: (start: number, end: number) => void;
  setData: (data: TimelinePoint[]) => void;
  tick: () => void;
}

const ADVANCE_PER_TICK_MS = 15 * 60 * 1000; // 15 minutes per tick at 1x speed

export const useTimelineStore = create<TimelineState>((set, get) => ({
  currentTime: Date.now(),
  isPlaying: false,
  playSpeed: 1,
  range: { start: Date.now() - 86400000, end: Date.now() },
  data: [],
  mode: "live",

  scrubTo: (timestamp: number) => set({ currentTime: timestamp }),

  play: () => set({ isPlaying: true }),

  pause: () => set({ isPlaying: false }),

  togglePlay: () => set((s) => ({ isPlaying: !s.isPlaying })),

  setSpeed: (speed: number) => set({ playSpeed: speed }),

  enterLiveMode: () =>
    set({
      mode: "live",
      isPlaying: false,
      range: { start: Date.now() - 86400000, end: Date.now() },
      currentTime: Date.now(),
    }),

  enterHistoricalMode: (start: number, end: number) =>
    set({
      mode: "historical",
      isPlaying: false,
      range: { start, end },
      currentTime: start,
    }),

  setData: (data: TimelinePoint[]) => set({ data }),

  tick: () => {
    const { isPlaying, playSpeed, currentTime, mode } = get();
    if (!isPlaying) return;
    const advanceMs = playSpeed * ADVANCE_PER_TICK_MS;
    const next = currentTime + advanceMs;
    const now = Date.now();
    // Clamp to not go beyond now in live mode
    if (mode === "live" && next > now) {
      set({ currentTime: now, isPlaying: false });
    } else {
      set({ currentTime: next });
    }
  },
}));