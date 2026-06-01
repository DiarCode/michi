import type { Station, TimelinePoint, TimelineMode } from "@/types";
import { MapMarker, MarkerContent, MarkerTooltip, MarkerPopup } from "@/components/ui/map";
import { LOAD_HIGH, LOAD_MID, STATION_CAPACITY, MORNING_PEAK, EVENING_PEAK } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface Props {
  station: Station;
  onClick?: (stationId: string) => void;
  hour?: number;
  predictedLoad?: number;
  /** Timeline mode determines marker appearance */
  timelineMode?: TimelineMode;
  /** Timeline data for this station at the selected time */
  timelineData?: TimelinePoint;
}

function getLoadPercent(station: Station, hour: number): number {
  const base = station.ridership_24h ?? 1000;
  if (hour >= MORNING_PEAK[0] && hour <= MORNING_PEAK[1]) return Math.min(95, Math.round(base / STATION_CAPACITY * 100));
  if (hour >= EVENING_PEAK[0] && hour >= EVENING_PEAK[1]) return Math.min(95, Math.round(base / STATION_CAPACITY * 100));
  if (hour >= 6 && hour <= 22) return Math.min(70, Math.round(base * 0.6 / STATION_CAPACITY * 100));
  return Math.min(30, Math.round(base * 0.15 / STATION_CAPACITY * 100));
}

/** Determine marker style based on timeline mode */
function getMarkerStyle(timelineMode?: TimelineMode, timelineData?: TimelinePoint) {
  if (!timelineMode || timelineMode === "live") {
    return { borderStyle: "solid", borderColor: "white", opacity: 1, badge: null };
  }

  // Historical mode: check if we have past actual data or future prediction
  if (timelineData) {
    if (timelineData.actual !== null) {
      // Past data — grey outline, solid fill
      return {
        borderStyle: "solid",
        borderColor: "#9ca3af",
        opacity: 0.8,
        badge: null as string | null,
      };
    }
    if (timelineData.predicted !== null) {
      // Future prediction — purple dashed outline
      return {
        borderStyle: "dashed",
        borderColor: "#9333ea",
        opacity: 1,
        badge: timelineData.confidence_upper !== null
          ? `${Math.round((1 - Math.abs(timelineData.confidence_upper - timelineData.predicted) / timelineData.predicted) * 100)}%`
          : null,
      };
    }
  }

  // Default fallback for historical mode without data
  return { borderStyle: "solid", borderColor: "#9ca3af", opacity: 0.5, badge: null };
}

export default function StationMarker({ station, onClick, hour = new Date().getHours(), predictedLoad, timelineMode, timelineData }: Props) {
  const load = getLoadPercent(station, hour);
  const color = load > LOAD_HIGH ? "#ef4444" : load > LOAD_MID ? "#f59e0b" : "#22c55e";
  const size = load > LOAD_HIGH ? 14 : load > LOAD_MID ? 11 : 8;
  const style = getMarkerStyle(timelineMode, timelineData);

  // Override color for historical modes
  const fillColor = timelineMode === "historical" && timelineData?.actual !== null && timelineData?.actual !== undefined
    ? "#9ca3af" // grey for past actual
    : timelineMode === "historical" && timelineData?.predicted !== null && timelineData?.predicted !== undefined
      ? "#9333ea" // purple for future prediction
      : color;

  return (
    <MapMarker longitude={station.lon} latitude={station.lat}>
      <MarkerContent>
        <div
          className={cn(
            "rounded-full border-2 shadow-md cursor-pointer transition-transform hover:scale-150",
          )}
          style={{
            backgroundColor: fillColor,
            width: size,
            height: size,
            borderColor: style.borderColor,
            borderStyle: style.borderStyle as "solid" | "dashed",
            opacity: style.opacity,
          }}
          onClick={() => onClick?.(station.id)}
        />
        {/* Confidence badge for future predictions */}
        {style.badge && (
          <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-purple-100 dark:bg-purple-900/60 text-purple-700 dark:text-purple-300 text-[7px] font-bold px-1 rounded">
            {style.badge}
          </div>
        )}
      </MarkerContent>
      <MarkerTooltip>
        <div className="text-xs">
          <div className="font-semibold">{station.name}</div>
          {timelineMode === "historical" && timelineData ? (
            <div className="text-gray-500">
              {timelineData.actual !== null ? (
                <span>Actual: {Math.round(timelineData.actual)} passengers</span>
              ) : timelineData.predicted !== null ? (
                <span>Forecast: {Math.round(timelineData.predicted)} passengers (range: {Math.round(timelineData.confidence_lower ?? timelineData.predicted)}–{Math.round(timelineData.confidence_upper ?? timelineData.predicted)})</span>
              ) : (
                <span>{station.ridership_24h?.toLocaleString() ?? "—"} passengers/day · {load}% capacity</span>
              )}
            </div>
          ) : (
            <div className="text-gray-500">{station.ridership_24h?.toLocaleString() ?? "—"} passengers/day · {load}% load</div>
          )}
        </div>
      </MarkerTooltip>
      <MarkerPopup>
        <div className="p-2 min-w-[180px]">
          <div className="font-bold text-sm">{station.name}</div>
          {station.district && <div className="text-xs text-gray-500">{station.district}</div>}

          {timelineMode === "historical" && timelineData ? (
            <>
              {timelineData.actual !== null && (
                <div className="mt-1 text-xs">
                  <span>Actual Passengers: </span>
                  <span className="font-mono font-bold text-gray-600">{Math.round(timelineData.actual)}</span>
                </div>
              )}
              {timelineData.predicted !== null && (
                <div className="mt-1 text-xs">
                  <span>Forecast: </span>
                  <span className="font-mono font-bold text-purple-600">{Math.round(timelineData.predicted)} passengers</span>
                </div>
              )}
              {timelineData.confidence_upper !== null && timelineData.confidence_lower !== null && (
                <div className="text-xs text-gray-400">
                  Range: {Math.round(timelineData.confidence_lower)} – {Math.round(timelineData.confidence_upper)} passengers
                </div>
              )}
            </>
          ) : (
            <>
              <div className="mt-1 text-xs">
                <span>Daily Passengers: </span>
                <span className="font-mono">{station.ridership_24h?.toLocaleString() ?? "—"}</span>
              </div>
              <div className="text-xs">
                <span>Capacity Used: </span>
                <span className="font-mono font-bold" style={{ color }}>{load}%</span>
              </div>
            </>
          )}

          {predictedLoad !== undefined && (
            <div className="text-xs mt-1 pt-1 border-t border-gray-200">
              <span>Forecast: </span>
              <span className="font-mono font-bold text-blue-600">{predictedLoad} passengers</span>
            </div>
          )}
        </div>
      </MarkerPopup>
    </MapMarker>
  );
}