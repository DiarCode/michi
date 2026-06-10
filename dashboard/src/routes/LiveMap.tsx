import { useState, useMemo } from "react";
import MapContainer from "@/components/map/MapContainer";
import TimelineBar from "@/components/map/TimelineBar";
import StationDetailPanel from "@/components/map/StationDetailPanel";
import { useStations } from "@/hooks/useStations";
import { useTimeline } from "@/hooks/useTimeline";
import { useRoutePaths } from "@/hooks/useRoutes";
import { fetchRoutes, fetchStationDetail } from "@/lib/api";
import { showToast } from "@/lib/toast";
import { useBusStore } from "@/stores/busStore";
import type { BusPosition, Route, StationDetail } from "@/types";
import { useQuery } from "@tanstack/react-query";
import { HugeiconsIcon } from "@hugeicons/react";
import { LayersIcon } from "@/lib/icons";

export default function LiveMap() {
  const { mode: timelineMode, currentTime, getStationData: getTimelineStationData } = useTimeline();

  const busPositions = useBusStore((s) => s.buses);
  const buses: BusPosition[] = useMemo(() => Object.values(busPositions) as unknown as BusPosition[], [busPositions]);

  // Fetch routes for coloring bus markers
  const { data: routesData } = useQuery({
    queryKey: ["routes"],
    queryFn: fetchRoutes,
    staleTime: 5 * 60 * 1000,
  });
  const routes: Route[] = routesData?.routes ?? [];

  const [selectedStation, setSelectedStation] = useState<StationDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showRouteLayers, setShowRouteLayers] = useState(false);

  // Use timeline hour for station data
  const timelineHour = useMemo(() => {
    if (timelineMode === "historical" && currentTime) {
      return new Date(currentTime).getHours();
    }
    return new Date().getHours();
  }, [timelineMode, currentTime]);

  const { data: stationData, isLoading } = useStations(timelineHour);
  const allStations = stationData?.stations ?? [];

  // Build route color map from routes data
  const routeColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const r of routes) {
      if (r.color) map[r.id] = r.color;
    }
    return map;
  }, [routes]);

  // Build station coordinate lookup for route path resolution
  const stationCoords = useMemo(() => {
    const map: Record<string, { lat: number; lon: number }> = {};
    for (const s of allStations) {
      map[s.id] = { lat: s.lat, lon: s.lon };
      map[s.name] = { lat: s.lat, lon: s.lon };
    }
    return map;
  }, [allStations]);

  // Fetch route paths for bus trails and layer toggle
  const { routePaths } = useRoutePaths(showRouteLayers ? routes : [], stationCoords);

  const handleStationClick = async (stationId: string) => {
    setLoading(true);
    try {
      setSelectedStation(await fetchStationDetail(stationId));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      showToast.error(`Failed to load station: ${message}`);
      setSelectedStation(null);
    } finally {
      setLoading(false);
    }
  };

  if (isLoading) {
    return <div className="h-[calc(100vh-4rem)] bg-background animate-pulse" />;
  }

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      <div className="flex-1 relative">
        <MapContainer
          stations={allStations}
          buses={buses}
          hour={timelineHour}
          onStationClick={handleStationClick}
          showHeatmap={showHeatmap}
          timelineMode={timelineMode}
          getTimelineStationData={getTimelineStationData}
          routeColorMap={routeColorMap}
          routes={showRouteLayers ? routes : []}
          routePaths={showRouteLayers ? routePaths : {}}
          showRouteLayers={showRouteLayers}
        />

        {/* Top-right controls */}
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className="px-4 py-2 text-sm font-semibold rounded-full transition-all bg-card/95 backdrop-blur-sm text-muted-foreground hover:bg-muted"
          >
            {showHeatmap ? "● Heatmap" : "○ Heatmap"}
          </button>
          <button
            onClick={() => setShowRouteLayers(!showRouteLayers)}
            className={`px-4 py-2 text-sm font-semibold rounded-full transition-all backdrop-blur-sm ${
              showRouteLayers
                ? "bg-primary text-primary-foreground"
                : "bg-card/95 text-muted-foreground hover:bg-muted"
            }`}
          >
            <HugeiconsIcon icon={LayersIcon} size={14} className="inline mr-1.5" />
            {showRouteLayers ? "Routes On" : "Routes"}
          </button>
        </div>

        {buses.length > 0 && (
          <div className="absolute top-4 left-4 z-10 px-4 py-2 text-sm font-semibold rounded-full bg-card/90 backdrop-blur-sm text-muted-foreground">
            <span className="inline-block w-2 h-2 rounded-full bg-chart-2 mr-2" />
            {buses.length} buses · {String(timelineHour).padStart(2, "0")}:00
          </div>
        )}

        {selectedStation && (
          <StationDetailPanel
            station={selectedStation}
            loading={loading}
            timelineMode={timelineMode}
            getTimelineStationData={getTimelineStationData}
            onClose={() => setSelectedStation(null)}
          />
        )}
      </div>

      <TimelineBar />
    </div>
  );
}