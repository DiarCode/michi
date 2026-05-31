import { useRef, useCallback, useEffect, useState } from "react";
import { useTimeline } from "@/hooks/useTimeline";
import type { PlaybackSpeed } from "@/types";
import { cn } from "@/lib/utils";

const SPEEDS: PlaybackSpeed[] = [1, 2, 5];

function formatTime(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDate(ms: number): string {
  const d = new Date(ms);
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function formatHourLabel(ms: number): string {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2, "0")}:00`;
}

export default function TimelineBar() {
  const {
    mode,
    currentTime,
    isPlaying,
    playSpeed,
    data,
    range,
    handleScrubStart,
    enterLiveMode,
    togglePlay,
    setSpeed,
    scrubTo,
  } = useTimeline();

  const barRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const now = Date.now();
  const rangeStart = range.start;
  const rangeEnd = range.end;
  const rangeDuration = rangeEnd - rangeStart;

  // Position of current time as fraction (0-1)
  const position = Math.max(0, Math.min(1, (currentTime - rangeStart) / rangeDuration));

  // Position of "now" as fraction
  const nowPosition = Math.max(0, Math.min(1, (now - rangeStart) / rangeDuration));

  // Compute hour markers
  const hourMarkers: { ms: number; label: string }[] = [];
  const firstHour = Math.ceil(rangeStart / (60 * 60 * 1000)) * (60 * 60 * 1000);
  for (let ms = firstHour; ms < rangeEnd; ms += 60 * 60 * 1000) {
    hourMarkers.push({ ms, label: formatHourLabel(ms) });
  }

  // Compute confidence band for future segment (aggregate across stations)
  const confidenceBand = (() => {
    if (mode !== "historical" || currentTime >= now) return null;
    // Get points near the current time in the future
    const futurePoints = data.filter(
      (p) =>
        new Date(p.timestamp).getTime() > currentTime &&
        new Date(p.timestamp).getTime() <= currentTime + 2 * 60 * 60 * 1000 &&
        p.predicted !== null &&
        p.confidence_upper !== null &&
        p.confidence_lower !== null
    );
    if (futurePoints.length === 0) return null;
    // Average confidence width as percentage of predicted
    let totalWidthPct = 0;
    let count = 0;
    for (const p of futurePoints) {
      if (p.predicted && p.predicted > 0) {
        const width = ((p.confidence_upper! - p.confidence_lower!) / p.predicted) * 100;
        totalWidthPct += width;
        count++;
      }
    }
    return count > 0 ? totalWidthPct / count : null;
  })();

  // Convert pixel position on bar to timestamp
  const pixelToTime = useCallback(
    (clientX: number): number => {
      if (!barRef.current) return currentTime;
      const rect = barRef.current.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return rangeStart + fraction * rangeDuration;
    },
    [rangeStart, rangeDuration, currentTime]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
      const time = pixelToTime(e.clientX);
      handleScrubStart(time);
    },
    [pixelToTime, handleScrubStart]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const time = pixelToTime(e.clientX);
      scrubTo(time);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, pixelToTime, scrubTo]);

  return (
    <div className="bg-white dark:bg-gray-900 border-t dark:border-gray-700 px-4 py-2 select-none">
      {/* Controls row */}
      <div className="flex items-center gap-3 mb-2">
        {/* Mode badge */}
        <span
          className={cn(
            "px-2 py-0.5 rounded text-xs font-bold tracking-wide",
            mode === "live"
              ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400"
              : "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-400"
          )}
        >
          {mode === "live" ? "LIVE" : "HISTORICAL"}
        </span>

        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" className="text-gray-700 dark:text-gray-300">
              <rect x="2" y="1" width="3.5" height="12" rx="1" />
              <rect x="8.5" y="1" width="3.5" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor" className="text-gray-700 dark:text-gray-300">
              <path d="M3 1.5L12 7L3 12.5Z" />
            </svg>
          )}
        </button>

        {/* Speed selector */}
        <div className="flex gap-1">
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={cn(
                "px-2 py-0.5 text-xs rounded font-mono font-medium transition-colors",
                playSpeed === s
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
              )}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Time display */}
        <div className="ml-auto flex items-center gap-2 text-sm">
          <span className="text-gray-400 dark:text-gray-500 text-xs">{formatDate(currentTime)}</span>
          <span className="font-mono font-semibold text-gray-800 dark:text-gray-200">{formatTime(currentTime)}</span>
        </div>

        {/* Return to Live button (only in historical mode) */}
        {mode === "historical" && (
          <button
            onClick={enterLiveMode}
            className="px-2 py-0.5 text-xs rounded font-medium bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-900/60 transition-colors"
          >
            Return to Live
          </button>
        )}
      </div>

      {/* Timeline bar */}
      <div
        ref={barRef}
        className="relative h-8 rounded cursor-pointer overflow-hidden"
        onMouseDown={handleMouseDown}
      >
        {/* Background segments */}
        <div className="absolute inset-0 flex">
          {/* Past: solid grey */}
          <div
            className="h-full bg-gray-300 dark:bg-gray-700"
            style={{ width: `${nowPosition * 100}%` }}
          />
          {/* Future: dashed purple */}
          <div
            className="h-full border-l border-gray-400 dark:border-gray-600"
            style={{
              width: `${(1 - nowPosition) * 100}%`,
              background: "repeating-linear-gradient(90deg, rgba(147,51,234,0.25) 0px, rgba(147,51,234,0.25) 6px, transparent 6px, transparent 12px)",
            }}
          />
        </div>

        {/* Confidence band on future segment */}
        {confidenceBand !== null && nowPosition < 1 && (
          <div
            className="absolute top-1 bottom-1 bg-purple-400/20 dark:bg-purple-500/20 rounded"
            style={{
              left: `${Math.max(position, nowPosition) * 100}%`,
              width: `${Math.max(0, (1 - Math.max(position, nowPosition))) * 100}%`,
            }}
          />
        )}

        {/* Hour markers */}
        {hourMarkers.map(({ ms, label }) => {
          const frac = (ms - rangeStart) / rangeDuration;
          if (frac < 0 || frac > 1) return null;
          return (
            <div
              key={ms}
              className="absolute top-0 h-full flex flex-col items-center"
              style={{ left: `${frac * 100}%` }}
            >
              <div className="w-px h-full bg-gray-400/30 dark:bg-gray-500/30" />
              <span className="absolute bottom-0 text-[8px] text-gray-400 dark:text-gray-500 -mb-0.5 transform -translate-x-1/2">
                {label}
              </span>
            </div>
          );
        })}

        {/* Now marker */}
        {nowPosition > 0 && nowPosition < 1 && (
          <div
            className="absolute top-0 h-full w-0.5 bg-blue-500 z-10"
            style={{ left: `${nowPosition * 100}%` }}
          >
            <span className="absolute -top-3 left-1/2 -translate-x-1/2 text-[8px] font-bold text-blue-600 dark:text-blue-400 whitespace-nowrap">
              NOW
            </span>
          </div>
        )}

        {/* Scrubber handle */}
        <div
          className={cn(
            "absolute top-0 h-full w-1 z-20",
            isDragging ? "w-1.5" : "w-1"
          )}
          style={{ left: `${position * 100}%`, transform: "translateX(-50%)" }}
        >
          {/* Vertical line */}
          <div className="w-full h-full bg-blue-600 dark:bg-blue-400 rounded-sm" />
          {/* Handle grip */}
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-1/2">
            <div className={cn(
              "w-4 h-6 rounded-sm border-2 border-blue-600 dark:border-blue-400 bg-white dark:bg-gray-900 shadow-md",
              isDragging && "scale-110"
            )} />
          </div>
        </div>
      </div>
    </div>
  );
}