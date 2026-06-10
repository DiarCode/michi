import type { Station, BusPosition, Route, TimelineMode, TimelinePoint } from "@/types";
import { Map, MapControls, MapClusterLayer } from "@/components/ui/map";
import StationMarker from "./StationMarker";
import AnimatedBusLayer from "./AnimatedBusLayer";
import BusTrail from "./BusTrail";
import RoutePathLayer from "./RoutePathLayer";
import ConnectionArcs from "./ConnectionArcs";
import { useThemeStore } from "@/stores/themeStore";
import { STATION_CAPACITY, MORNING_PEAK, EVENING_PEAK } from "@/lib/constants";

interface Props {
  stations: Station[];
  buses: BusPosition[];
  hour?: number;
  onStationClick?: (stationId: string) => void;
  showHeatmap?: boolean;
  /** Timeline mode — controls marker styling and data source */
  timelineMode?: TimelineMode;
  /** Get timeline data for a station at the selected time */
  getTimelineStationData?: (stationId: string) => TimelinePoint | undefined;
  /** Route ID → color map for coloring bus markers by route */
  routeColorMap?: Record<string, string>;
  /** Routes data for drawing path lines (only when layers enabled) */
  routes?: Route[];
  /** Route ID → [lon, lat][] polyline coordinates (only when layers enabled) */
  routePaths?: Record<string, [number, number][]>;
  /** Whether to show route layers (paths + connection arcs) */
  showRouteLayers?: boolean;
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

export default function MapContainer({
  stations,
  buses,
  hour = new Date().getHours(),
  onStationClick,
  showHeatmap = true,
  timelineMode,
  getTimelineStationData,
  routeColorMap = {},
  routes = [],
  routePaths = {},
  showRouteLayers = false,
}: Props) {
  const resolvedTheme = useThemeStore((s) => s.resolvedTheme);
  const clusterData = showHeatmap
    ? buildClusterData(stations, hour, timelineMode, getTimelineStationData)
    : null;

  return (
    <div className="relative w-full h-full">
      <Map
        center={ASTANA_CENTER}
        zoom={ASTANA_ZOOM}
        theme={resolvedTheme === "dark" ? "dark" : "light"}
        className="w-full h-full rounded-lg overflow-hidden ring-1 ring-foreground/5"
      >
        <MapControls showZoom showCompass />

        {/* Route layers — only rendered when toggle is on */}
        {showRouteLayers && routes.length > 0 && Object.keys(routePaths).length > 0 && (
          <RoutePathLayer
            routes={routes}
            routePaths={routePaths}
            routeColorMap={routeColorMap}
            highlightedRouteId={null}
          />
        )}

        {clusterData && <MapClusterLayer data={clusterData} clusterRadius={50} clusterMaxZoom={15} />}

        {stations.map((s) => {
          const td = getTimelineStationData?.(s.id);
          return (
            <StationMarker
              key={s.id}
              station={s}
              onClick={onStationClick}
              hour={hour}
              timelineMode={timelineMode}
              timelineData={td}
            />
          );
        })}

        {buses.length > 0 && (
          <>
            <BusTrail routeColorMap={routeColorMap} />
            <AnimatedBusLayer buses={buses} routeColorMap={routeColorMap} routePaths={routePaths} />
            {showRouteLayers && <ConnectionArcs />}
          </>
        )}
      </Map>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-card/90 backdrop-blur-sm p-2.5 rounded-lg text-xs space-y-1 z-10 ring-1 ring-foreground/5">
        <div className="font-semibold text-foreground">Capacity Used</div>
        <div className="flex items-center gap-1.5 text-muted-foreground"><span className="w-3 h-3 rounded-full bg-chart-2" /> Low (&lt;50%)</div>
        <div className="flex items-center gap-1.5 text-muted-foreground"><span className="w-3 h-3 rounded-full bg-chart-4" /> Medium (50–80%)</div>
        <div className="flex items-center gap-1.5 text-muted-foreground"><span className="w-3 h-3 rounded-full bg-destructive" /> High (&gt;80%)</div>
        {timelineMode === "historical" && (
          <div className="border-t border-border pt-1 mt-1 space-y-1">
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <span className="w-3 h-3 rounded-full border-2 border-muted-foreground" /> Past (actual)
            </div>
            <div className="flex items-center gap-1.5 text-muted-foreground">
              <span className="w-3 h-3 rounded-full border-2 border-dashed border-primary bg-primary/30" /> Future (predicted)
            </div>
          </div>
        )}
      </div>

      {/* Info overlay */}
      <div className="absolute top-4 left-4 bg-card/90 backdrop-blur-sm p-3 rounded-lg z-10 ring-1 ring-foreground/5">
        <h3 className="font-bold text-sm text-foreground">
          {timelineMode === "historical" ? "Historical View" : "Live Tracking"}
        </h3>
        <p className="text-xs text-muted-foreground">
          {buses.length} buses · {String(hour).padStart(2, "0")}:00
        </p>
      </div>
    </div>
  );
}