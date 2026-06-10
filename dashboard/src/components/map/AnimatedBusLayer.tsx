import { useEffect, useRef } from "react";
import { MapMarker, MarkerContent, MarkerPopup } from "@/components/ui/map";
import { useAnimationStore } from "@/stores/animationStore";
import type { BusPosition } from "@/types";

/** Occupancy color: green (low) → amber (medium) → red (high) */
function occupancyColor(pct: number): string {
  if (pct >= 80) return "var(--destructive)";
  if (pct >= 50) return "var(--chart-4)";
  return "var(--chart-2)";
}

interface AnimatedBusLayerProps {
  buses: BusPosition[];
  routeColorMap: Record<string, string>;
  routePaths?: Record<string, [number, number][]>;
}

export default function AnimatedBusLayer({ buses, routeColorMap }: AnimatedBusLayerProps) {
  const { interpolatedPositions, start, stop, updateTargets } = useAnimationStore();
  const startedRef = useRef(false);

  // Start animation loop on mount, stop on unmount
  useEffect(() => {
    if (!startedRef.current) {
      start();
      startedRef.current = true;
    }
    return () => {
      stop();
      startedRef.current = false;
    };
  }, [start, stop]);

  // Feed new bus positions as targets for interpolation
  useEffect(() => {
    updateTargets(buses);
  }, [buses, updateTargets]);

  const interpolated = Object.values(interpolatedPositions);

  return (
    <>
      {interpolated.map((bus) => {
        const routeColor = routeColorMap[bus.route_id] ?? "#888";
        const occColor = occupancyColor(bus.occupancy_percent);

        return (
          <MapMarker
            key={bus.bus_id}
            longitude={bus.lon}
            latitude={bus.lat}
          >
            <MarkerContent>
              <div
                className="relative flex items-center justify-center"
                style={{ transform: `rotate(${bus.bearing}deg)` }}
              >
                {/* Bus direction arrow */}
                <div
                  className="w-5 h-5 rounded-full border-2 flex items-center justify-center"
                  style={{
                    backgroundColor: routeColor,
                    borderColor: occColor,
                    transform: "rotate(0deg)",
                  }}
                >
                  {/* Direction indicator */}
                  <svg
                    width="10"
                    height="10"
                    viewBox="0 0 10 10"
                    className="text-white"
                    style={{ transform: "rotate(0deg)" }}
                  >
                    <polygon points="5,1 9,8 1,8" fill="white" opacity={0.9} />
                  </svg>
                </div>
              </div>
            </MarkerContent>
            <MarkerPopup>
              <div className="space-y-1 text-xs min-w-[140px]">
                <div className="font-bold text-sm">{bus.bus_id}</div>
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: routeColor }}
                  />
                  <span className="font-medium">{bus.route_id}</span>
                </div>
                {bus.speed_kmh != null && (
                  <div className="text-muted-foreground">{bus.speed_kmh} km/h</div>
                )}
                {bus.occupancy_percent != null && (
                  <div className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: occColor }}
                    />
                    <span>{bus.occupancy_percent}% full</span>
                  </div>
                )}
                {bus.next_stop && (
                  <div className="text-muted-foreground">
                    Next: {bus.next_stop}
                    {bus.eta_seconds != null && ` · ${Math.ceil(bus.eta_seconds / 60)}m`}
                  </div>
                )}
              </div>
            </MarkerPopup>
          </MapMarker>
        );
      })}
    </>
  );
}