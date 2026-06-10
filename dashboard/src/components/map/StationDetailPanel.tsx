import { useEffect, useRef } from "react";
import type { StationDetail } from "@/types";
import type { TimelinePoint } from "@/types";
import ConfidenceBadge from "@/components/ui/ConfidenceBadge";

interface StationDetailPanelProps {
  station: StationDetail;
  loading?: boolean;
  timelineMode?: "live" | "historical";
  getTimelineStationData?: (stationId: string) => TimelinePoint | null | undefined;
  onClose: () => void;
}

const severityColor: Record<string, string> = {
  critical: "border-l-destructive bg-destructive/10 text-destructive",
  warning: "border-l-chart-4 bg-chart-4/10 text-chart-4",
  info: "border-l-primary bg-primary/10 text-primary",
};

export default function StationDetailPanel({
  station,
  loading = false,
  timelineMode = "live",
  getTimelineStationData,
  onClose,
}: StationDetailPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Animate slide-in on mount
  useEffect(() => {
    const el = panelRef.current;
    if (!el) return;
    el.style.transform = "translateX(100%)";
    el.style.transition = "transform 200ms ease-out";
    requestAnimationFrame(() => {
      el.style.transform = "translateX(0)";
    });
  }, []);

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const { station: s, connected_routes, forecast, alerts, hourly_ridership } = station;

  const maxRidership = Math.max(...hourly_ridership.map((h) => h.ridership), 1);

  return (
    <div
      ref={panelRef}
      className="absolute top-0 right-0 w-80 h-full bg-card/95 backdrop-blur-sm shadow-lg overflow-y-auto border-l border-border ring-1 ring-foreground/5"
      style={{ transform: "translateX(100%)" }}
    >
      {/* Header */}
      <div className="sticky top-0 bg-card/90 backdrop-blur-sm z-10 px-5 pt-4 pb-3 border-b border-border">
        <button
          onClick={onClose}
          className="absolute top-3 right-3 w-7 h-7 rounded-full bg-muted flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-border transition-colors text-sm"
          aria-label="Close station details"
        >
          ✕
        </button>
        <h3 className="font-bold text-lg text-foreground pr-8">{s.name}</h3>
        <p className="text-sm text-muted-foreground mt-0.5">
          {s.district ?? "Unknown district"} · {(s.ridership_24h ?? 0).toLocaleString()} passengers/day
        </p>
        {s.load_percent != null && (
          <div className="mt-2 flex items-center gap-2">
            <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${Math.min(s.load_percent, 100)}%`,
                  backgroundColor: s.load_percent > 90 ? "var(--destructive)" : s.load_percent > 70 ? "var(--chart-4)" : "var(--chart-2)",
                }}
              />
            </div>
            <span className="text-xs font-mono text-muted-foreground">{s.load_percent.toFixed(0)}%</span>
          </div>
        )}
      </div>

      <div className="px-5 pb-5">
        {/* Timeline Data (historical mode only) */}
        {timelineMode === "historical" && getTimelineStationData && (() => {
          const td = getTimelineStationData(s.id);
          if (!td) return null;
          return (
            <div className="mt-4 p-3 bg-chart-2/10 rounded-xl text-sm">
              <div className="font-semibold text-chart-2 mb-1.5">Timeline Data</div>
              {td.actual !== null && (
                <div className="text-muted-foreground">
                  Actual: <span className="font-mono font-bold">{Math.round(td.actual)} passengers</span>
                </div>
              )}
              {td.predicted !== null && (
                <div className="text-chart-1">
                  Forecast: <span className="font-mono font-bold">{Math.round(td.predicted)} passengers</span>
                  {td.confidence_upper !== null && td.confidence_lower !== null && (
                    <span className="text-muted-foreground ml-1">
                      (range: {Math.round(td.confidence_lower)}–{Math.round(td.confidence_upper)})
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })()}

        {/* Connected Routes */}
        {connected_routes.length > 0 && (
          <div className="mt-4">
            <h4 className="font-semibold text-sm text-foreground mb-2">Connected Routes</h4>
            <div className="flex gap-2 flex-wrap">
              {connected_routes.map((r) => (
                <button
                  key={r.id}
                  className="px-3 py-1 text-xs rounded-full text-white font-semibold hover:opacity-80 transition-opacity"
                  style={{ backgroundColor: r.color ?? "#888" }}
                >
                  {r.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Hourly Ridership Chart */}
        {hourly_ridership.length > 0 && (
          <div className="mt-4">
            <h4 className="font-semibold text-sm text-foreground mb-2">Hourly Pattern</h4>
            <div className="flex items-end gap-px h-20 mt-1">
              {hourly_ridership.map((h) => {
                const pct = (h.ridership / maxRidership) * 100;
                const isRush = (h.hour >= 7 && h.hour <= 9) || (h.hour >= 17 && h.hour <= 19);
                return (
                  <div key={h.hour} className="flex-1 flex flex-col justify-end group relative">
                    <div className="hidden group-hover:block absolute -top-6 left-1/2 -translate-x-1/2 text-[9px] bg-foreground text-background px-1.5 py-0.5 rounded whitespace-nowrap z-10">
                      {h.hour}:00 — {h.ridership.toLocaleString()}
                    </div>
                    <div
                      className={`${isRush ? "bg-chart-2" : "bg-border"} rounded-sm transition-colors hover:bg-chart-1`}
                      style={{ height: `${pct}%`, minHeight: 2 }}
                    />
                  </div>
                );
              })}
            </div>
            <div className="flex justify-between text-[9px] text-muted-foreground mt-1">
              <span>0</span><span>6</span><span>12</span><span>18</span><span>23</span>
            </div>
          </div>
        )}

        {/* Forecast Table */}
        {forecast.length > 0 && (
          <div className="mt-4">
            <h4 className="font-semibold text-sm text-foreground mb-2">Forecast (next 6 hours)</h4>
            <div className="space-y-1.5 mt-1">
              {forecast.slice(0, 6).map((f, i) => (
                <div key={i} className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>{new Date(f.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                  <span className="font-mono font-semibold">{f.predicted} passengers</span>
                  <ConfidenceBadge confidence={f.confidence} compact />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Active Alerts */}
        {alerts.length > 0 && (
          <div className="mt-4">
            <h4 className="font-semibold text-sm text-destructive mb-2">Active Alerts</h4>
            {alerts.map((a, i) => (
              <div
                key={i}
                className={`mt-1.5 p-2.5 border-l-4 rounded-xl text-sm ${
                  severityColor[a.severity] ?? severityColor.info
                }`}
              >
                <span className="font-semibold uppercase text-xs">{a.severity}</span>: {a.title}
              </div>
            ))}
          </div>
        )}

        {loading && (
          <div className="mt-4 flex items-center gap-2 text-sm text-muted-foreground">
            <span className="animate-spin inline-block w-4 h-4 border-2 border-chart-2 border-t-transparent rounded-full" />
            Loading details...
          </div>
        )}
      </div>
    </div>
  );
}