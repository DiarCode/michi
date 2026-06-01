import type { Station, BusPosition, PredictionPoint, TimelineMode, TimelinePoint } from "@/types";
import { Map, MapControls, MapClusterLayer } from "@/components/ui/map";
import StationMarker from "./StationMarker";
import BusMarker from "./BusMarker";
import { STATION_CAPACITY, MORNING_PEAK, EVENING_PEAK } from "@/lib/constants";

interface Props {
  stations: Station[];
  buses: BusPosition[];
  hour?: number;
  onStationClick?: (stationId: string) => void;
  showHeatmap?: boolean;
  predictions?: PredictionPoint[];
  /** Timeline mode — controls marker styling and data source */
  timelineMode?: TimelineMode;
  /** Get timeline data for a station at the selected time */
  getTimelineStationData?: (stationId: string) => TimelinePoint | undefined;
  /** Route ID → color map for coloring bus markers by route */
  routeColorMap?: Record<string, string>;
}

const ASTANA_CENTER: [number, number] = [71.47, 51.13];
const ASTANA_ZOOM = 11;

function getLoadPercent(s: Station, hour: number): number {
  const base = s.ridership_24h ?? 1000;
  if ((hour >= MORNING_PEAK[0] && hour <= MORNING_PEAK[1]) || (hour >= EVENING_PEAK[0] && hour <= EVENING_PEAK[1])) {
    return Math.min(95, Math.round(base / STATION_CAPACITY * 100));
  }
  if (hour >= 6 && hour <= 22) return Math.min(70, Math.round(base * 0.6 / STATION_CAPACITY * 100));
  return Math.min(30, Math.round(base * 0.15 / STATION_CAPACITY * 100));
}

function getLoadFromTimelineData(td: TimelinePoint | undefined, fallback: number): number {
  if (!td) return fallback;
  const value = td.actual ?? td.predicted;
  if (value === null || value === undefined) return fallback;
  return Math.min(100, Math.round((value / STATION_CAPACITY) * 100));
}

function buildClusterData(
  stations: Station[],
  hour: number,
  timelineMode?: TimelineMode,
  getTimelineStationData?: (stationId: string) => TimelinePoint | undefined,
): GeoJSON.FeatureCollection<GeoJSON.Point> {
  return {
    type: "FeatureCollection",
    features: stations.map((s) => {
      const fallbackLoad = getLoadPercent(s, hour);
      const td = getTimelineStationData?.(s.id);
      const load = timelineMode === "historical" && td
        ? getLoadFromTimelineData(td, fallbackLoad)
        : fallbackLoad;

      return {
        type: "Feature" as const,
        geometry: {
          type: "Point" as const,
          coordinates: [s.lon, s.lat],
        },
        properties: {
          id: s.id,
          name: s.name,
          load,
          ridership: s.ridership_24h ?? 0,
        },
      };
    }),
  };
}

function buildPredictionLookup(predictions: PredictionPoint[]): Record<string, PredictionPoint> {
  const map: Record<string, PredictionPoint> = {};
  for (const p of predictions) {
    map[p.station_id] = p;
  }
  return map;
}

export default function MapContainer({
  stations,
  buses,
  hour = new Date().getHours(),
  onStationClick,
  showHeatmap = true,
  predictions = [],
  timelineMode,
  getTimelineStationData,
  routeColorMap = {},
}: Props) {
  const clusterData = showHeatmap
    ? buildClusterData(stations, hour, timelineMode, getTimelineStationData)
    : null;
  const predMap = buildPredictionLookup(predictions);

  return (
    <div className="relative w-full h-full">
      <Map
        center={ASTANA_CENTER}
        zoom={ASTANA_ZOOM}
        className="w-full h-full rounded-lg overflow-hidden"
      >
        <MapControls showZoom showCompass />

        {clusterData && <MapClusterLayer data={clusterData} clusterRadius={50} clusterMaxZoom={15} />}

        {stations.map((s) => {
          const pred = predMap[s.id];
          const td = getTimelineStationData?.(s.id);
          return (
            <StationMarker
              key={s.id}
              station={s}
              onClick={onStationClick}
              hour={hour}
              predictedLoad={pred ? Math.round(pred.predicted) : undefined}
              timelineMode={timelineMode}
              timelineData={td}
            />
          );
        })}

        {buses.map((b) => (
          <BusMarker key={b.bus_id} bus={b} routeColor={routeColorMap[b.route_id]} />
        ))}
      </Map>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm p-2 rounded-lg shadow-md text-xs space-y-1 z-10">
        <div className="font-semibold text-gray-700 dark:text-gray-300">Capacity Used</div>
        <div className="flex items-center gap-1.5 dark:text-gray-300"><span className="w-3 h-3 rounded-full bg-green-500" /> Low (&lt;50%)</div>
        <div className="flex items-center gap-1.5 dark:text-gray-300"><span className="w-3 h-3 rounded-full bg-amber-500" /> Medium (50–80%)</div>
        <div className="flex items-center gap-1.5 dark:text-gray-300"><span className="w-3 h-3 rounded-full bg-red-500" /> High (&gt;80%)</div>
        {timelineMode === "historical" && (
          <div className="border-t dark:border-gray-700 pt-1 mt-1 space-y-1">
            <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
              <span className="w-3 h-3 rounded-full border-2 border-gray-400" /> Past (actual)
            </div>
            <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
              <span className="w-3 h-3 rounded-full border-2 border-dashed border-purple-600 bg-purple-400/30" /> Future (predicted)
            </div>
          </div>
        )}
        {predictions.length > 0 && (
          <div className="border-t dark:border-gray-700 pt-1 mt-1 text-gray-500 dark:text-gray-400">
            Showing +{predictions[0]?.horizon_minutes ?? 0}m predictions
          </div>
        )}
      </div>

      {/* Info overlay */}
      <div className="absolute top-4 left-4 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm p-3 rounded-lg shadow-md z-10">
        <h3 className="font-bold text-sm text-gray-800 dark:text-white">
          {timelineMode === "historical" ? "Historical View" : "Live Tracking"}
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {buses.length} buses · {String(hour).padStart(2, "0")}:00
        </p>
      </div>
    </div>
  );
}