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

function formatRelativeLabel(offsetHours: number): string {
  if (offsetHours === 0) return "NOW";
  if (offsetHours > 0) return `+${offsetHours}h`;
  return `${offsetHours}h`;
}

export default function TimelineBar() {
  const {
    mode,
    currentTime,
    isPlaying,
    playSpeed,
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

  // Generate hour markers labeled relative to NOW
  const hourMarkers: { ms: number; offsetHours: number; isMajor: boolean }[] = [];
  const firstHour = Math.ceil(rangeStart / (60 * 60 * 1000)) * (60 * 60 * 1000);
  for (let ms = firstHour; ms < rangeEnd; ms += 60 * 60 * 1000) {
    const offsetHours = Math.round((ms - now) / (60 * 60 * 1000));
    const isMajor = offsetHours % 3 === 0;
    hourMarkers.push({ ms, offsetHours, isMajor });
  }

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
    <div className="bg-background border-t border-transparent px-4 pt-3 pb-3 select-none">
      {/* Controls row */}
      <div className="flex items-center gap-3 mb-3">
        {/* Mode badge */}
        <span
          className={cn(
            "px-3.5 py-1.5 rounded-full text-xs font-bold tracking-wide",
            mode === "live"
              ? "bg-chart-2/15 text-chart-2"
              : "bg-chart-4/15 text-chart-4"
          )}
        >
          {mode === "live" ? "● LIVE" : "◉ HISTORICAL"}
        </span>

        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="w-9 h-9 flex items-center justify-center rounded-full bg-muted hover:bg-muted/80 transition-colors"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg width="16" height="16" viewBox="0 0 14 14" fill="currentColor" className="text-foreground">
              <rect x="2" y="1" width="3.5" height="12" rx="1" />
              <rect x="8.5" y="1" width="3.5" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 14 14" fill="currentColor" className="text-foreground">
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
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              )}
            >
              {s}×
            </button>
          ))}
        </div>

        {/* Time display */}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-muted-foreground text-sm font-medium">{formatDate(currentTime)}</span>
          <span className="font-mono text-lg font-bold text-foreground tabular-nums">{formatTime(currentTime)}</span>
        </div>
      </div>

      {/* Timeline bar */}
      <div
        ref={barRef}
        className="relative h-12 rounded-lg cursor-pointer overflow-hidden bg-muted"
        onMouseDown={handleMouseDown}
      >
        {/* Past segment background (left of NOW) */}
        <div className="absolute inset-0 bg-muted/50" />

        {/* Future segment (right of NOW) — subtle pattern */}
        {nowPosition > 0 && nowPosition < 1 && (
          <div
            className="absolute top-0 bottom-0"
            style={{
              left: `${nowPosition * 100}%`,
              width: `${(1 - nowPosition) * 100}%`,
              background: "repeating-linear-gradient(90deg, rgba(139,92,246,0.08) 0px, rgba(139,92,246,0.08) 6px, transparent 6px, transparent 12px)",
            }}
          />
        )}

        {/* Hour markers with relative labels */}
        {hourMarkers.map(({ ms, offsetHours, isMajor }) => {
          const frac = (ms - rangeStart) / rangeDuration;
          if (frac < 0 || frac > 1) return null;
          const isNow = offsetHours === 0;
          return (
            <div
              key={ms}
              className="absolute top-0 h-full flex flex-col items-center"
              style={{ left: `${frac * 100}%` }}
            >
              <div className={cn(
                "w-px h-full",
                isNow ? "" : isMajor ? "bg-border" : "bg-border/30"
              )} />
              {isMajor && (
                <span className={cn(
                  "absolute bottom-1 transform -translate-x-1/2 whitespace-nowrap text-[10px]",
                  offsetHours < 0
                    ? "font-medium text-muted-foreground"
                    : offsetHours === 0
                      ? "font-bold text-primary"
                      : "font-medium text-chart-5"
                )}>
                  {formatRelativeLabel(offsetHours)}
                </span>
              )}
            </div>
          );
        })}

        {/* NOW marker — prominent blue vertical line, clickable to return to live */}
        {nowPosition > 0 && nowPosition < 1 && (
          <div
            className="absolute top-0 h-full w-0.5 bg-primary z-10"
            style={{ left: `${nowPosition * 100}%` }}
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => { e.stopPropagation(); enterLiveMode(); }}
          >
            <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold text-primary-foreground bg-primary px-2 py-0.5 rounded-full whitespace-nowrap cursor-pointer hover:bg-primary/90 transition-colors">
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
          <div className="w-full h-full bg-foreground rounded-sm" />
          {/* Handle grip */}
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-1/2">
            <div className={cn(
              "w-5 h-8 rounded border-2 border-foreground bg-background shadow-lg transition-transform",
              isDragging && "scale-110"
            )} />
          </div>
        </div>
      </div>
    </div>
  );
}