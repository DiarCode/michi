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

  const position = Math.max(0, Math.min(1, (currentTime - rangeStart) / rangeDuration));
  const nowPosition = Math.max(0, Math.min(1, (now - rangeStart) / rangeDuration));

  const hourMarkers: { ms: number; label: string; isMajor: boolean }[] = [];
  const firstHour = Math.ceil(rangeStart / (60 * 60 * 1000)) * (60 * 60 * 1000);
  for (let ms = firstHour; ms < rangeEnd; ms += 60 * 60 * 1000) {
    const h = new Date(ms).getHours();
    const isMajor = h % 6 === 0;
    hourMarkers.push({ ms, label: formatHourLabel(ms), isMajor });
  }

  const confidenceBand = (() => {
    if (mode !== "historical" || currentTime >= now) return null;
    const futurePoints = data.filter(
      (p) =>
        new Date(p.timestamp).getTime() > currentTime &&
        new Date(p.timestamp).getTime() <= currentTime + 2 * 60 * 60 * 1000 &&
        p.predicted !== null &&
        p.confidence_upper !== null &&
        p.confidence_lower !== null
    );
    if (futurePoints.length === 0) return null;
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
    const handleMouseUp = () => { setIsDragging(false); };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, pixelToTime, scrubTo]);

  return (
    <div className="bg-white border-t border-michi-border px-4 pt-3 pb-3 select-none">
      {/* Controls row */}
      <div className="flex items-center gap-3 mb-3">
        {/* Mode badge */}
        <span
          className={cn(
            "px-3.5 py-1.5 rounded-full text-xs font-bold tracking-wide",
            mode === "live"
              ? "bg-michi-lime/15 text-michi-lime-dark"
              : "bg-michi-amber/15 text-michi-amber"
          )}
        >
          {mode === "live" ? "● LIVE" : "◉ HISTORICAL"}
        </span>

        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-michi-warm hover:bg-michi-border transition-colors"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg width="16" height="16" viewBox="0 0 14 14" fill="currentColor" className="text-michi-dark">
              <rect x="2" y="1" width="3.5" height="12" rx="1" />
              <rect x="8.5" y="1" width="3.5" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 14 14" fill="currentColor" className="text-michi-dark">
              <path d="M3 1.5L12 7L3 12.5Z" />
            </svg>
          )}
        </button>

        {/* Speed selector */}
        <div className="flex gap-1.5">
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={cn(
                "px-3 py-1 text-xs rounded-full font-mono font-semibold transition-all",
                playSpeed === s
                  ? "bg-michi-dark text-white shadow-sm"
                  : "bg-michi-warm text-michi-muted border border-michi-border hover:bg-michi-border"
              )}
            >
              {s}×
            </button>
          ))}
        </div>

        {/* Time display */}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-michi-muted text-sm font-medium">{formatDate(currentTime)}</span>
          <span className="font-mono text-lg font-bold text-michi-dark tabular-nums">{formatTime(currentTime)}</span>
        </div>

        {/* Return to Live button */}
        {mode === "historical" && (
          <button
            onClick={enterLiveMode}
            className="px-3.5 py-1.5 text-xs rounded-full font-semibold bg-michi-lime/15 text-michi-lime-dark hover:bg-michi-lime/25 transition-colors"
          >
            Return to Live
          </button>
        )}
      </div>

      {/* Timeline bar */}
      <div
        ref={barRef}
        className="relative h-12 rounded-lg cursor-pointer overflow-hidden shadow-inner"
        onMouseDown={handleMouseDown}
      >
        {/* Background segments */}
        <div className="absolute inset-0 flex">
          <div className="h-full bg-michi-border" style={{ width: `${nowPosition * 100}%` }} />
          <div
            className="h-full border-l border-michi-muted"
            style={{
              width: `${(1 - nowPosition) * 100}%`,
              background: "repeating-linear-gradient(90deg, rgba(139,92,246,0.15) 0px, rgba(139,92,246,0.15) 6px, transparent 6px, transparent 12px)",
            }}
          />
        </div>

        {/* Confidence band on future segment */}
        {confidenceBand !== null && nowPosition < 1 && (
          <div
            className="absolute top-1 bottom-1 bg-michi-purple/15 rounded"
            style={{
              left: `${Math.max(position, nowPosition) * 100}%`,
              width: `${Math.max(0, (1 - Math.max(position, nowPosition))) * 100}%`,
            }}
          />
        )}

        {/* Hour markers with labels */}
        {hourMarkers.map(({ ms, label, isMajor }) => {
          const frac = (ms - rangeStart) / rangeDuration;
          if (frac < 0 || frac > 1) return null;
          return (
            <div
              key={ms}
              className="absolute top-0 h-full flex flex-col items-center"
              style={{ left: `${frac * 100}%` }}
            >
              <div className={cn(
                "w-px h-full",
                isMajor ? "bg-michi-muted/50" : "bg-michi-border"
              )} />
              <span className={cn(
                "absolute bottom-1 transform -translate-x-1/2 whitespace-nowrap",
                isMajor
                  ? "text-[10px] font-semibold text-michi-body"
                  : "text-[8px] text-michi-muted"
              )}>
                {label}
              </span>
            </div>
          );
        })}

        {/* Now marker */}
        {nowPosition > 0 && nowPosition < 1 && (
          <div
            className="absolute top-0 h-full w-0.5 bg-michi-lime z-10"
            style={{ left: `${nowPosition * 100}%` }}
          >
            <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold text-michi-dark whitespace-nowrap bg-michi-lime/15 px-2 py-0.5 rounded-full">
              NOW
            </span>
          </div>
        )}

        {/* Scrubber handle */}
        <div
          className={cn(
            "absolute top-0 h-full z-20 transition-[width] duration-75",
            isDragging ? "w-2" : "w-1.5"
          )}
          style={{ left: `${position * 100}%`, transform: "translateX(-50%)" }}
        >
          {/* Vertical line */}
          <div className="w-full h-full bg-michi-dark rounded-sm" />
          {/* Handle grip */}
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-1/2">
            <div className={cn(
              "w-5 h-8 rounded border-2 border-michi-dark bg-white shadow-lg transition-transform",
              isDragging && "scale-110"
            )} />
          </div>
        </div>
      </div>
    </div>
  );
}