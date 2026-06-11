import { useEffect, useState } from "react"
import type * as GeoJSON from "geojson"
import {
  useMap,
  MapClusterLayer,
} from "@/components/ui/map"
import StationMarker from "./StationMarker"
import type {
  Station,
  PredictionPoint,
  TimelineMode,
  TimelinePoint,
} from "@/types"

interface Props {
  stations: Station[]
  hour: number
  predictions: PredictionPoint[]
  timelineMode?: TimelineMode
  onStationClick?: (stationId: string) => void
  getTimelineStationData?: (stationId: string) => TimelinePoint | undefined
  showHeatmap: boolean
}

const CLUSTER_MAX_ZOOM = 14
const INDIVIDUAL_MARKER_MIN_ZOOM = 14

function buildClusterData(
  stations: Station[],
  hour: number,
  timelineMode?: TimelineMode,
  getTimelineStationData?: (stationId: string) => TimelinePoint | undefined
): GeoJSON.FeatureCollection<GeoJSON.Point> {
  return {
    type: "FeatureCollection",
    features: stations.map((s) => {
      const base = s.ridership_24h ?? 1000
      let load = 30
      if (hour >= 7 && hour <= 9) load = Math.min(95, Math.round((base / 2000) * 100))
      else if (hour >= 17 && hour <= 19) load = Math.min(95, Math.round((base / 2000) * 100))
      else if (hour >= 6 && hour <= 22) load = Math.min(70, Math.round(((base * 0.6) / 2000) * 100))
      else load = Math.min(30, Math.round(((base * 0.15) / 2000) * 100))

      const td = getTimelineStationData?.(s.id)
      if (timelineMode === "historical" && td) {
        const v = td.actual ?? td.predicted
        if (v != null) load = Math.min(100, Math.round((v / 2000) * 100))
      }

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
      }
    }),
  }
}

function buildPredictionLookup(
  predictions: PredictionPoint[]
): Record<string, PredictionPoint> {
  const map: Record<string, PredictionPoint> = {}
  for (const p of predictions) {
    map[p.station_id] = p
  }
  return map
}

/**
 * Renders stations using a hybrid approach:
 *  - At low zoom (≤ 14): a single MapClusterLayer handles all stations efficiently.
 *  - At high zoom (> 14): individual clickable StationMarkers are rendered.
 *
 * This avoids creating 374 React-managed MapLibre markers when the user is
 * looking at the city at a glance, where clusters are sufficient.
 */
export default function ZoomAwareStations({
  stations,
  hour,
  predictions,
  timelineMode,
  onStationClick,
  getTimelineStationData,
  showHeatmap,
}: Props) {
  const { map, isLoaded } = useMap()
  const [zoom, setZoom] = useState<number>(11)
  const [viewportBbox, setViewportBbox] = useState<{
    minLat: number
    maxLat: number
    minLon: number
    maxLon: number
  } | null>(null)

  // Track current zoom and visible bounding box.
  useEffect(() => {
    if (!map) return
    const update = () => {
      const z = map.getZoom()
      setZoom(z)
      if (z > INDIVIDUAL_MARKER_MIN_ZOOM) {
        const b = map.getBounds()
        setViewportBbox({
          minLat: b.getSouth(),
          maxLat: b.getNorth(),
          minLon: b.getWest(),
          maxLon: b.getEast(),
        })
      } else {
        setViewportBbox(null)
      }
    }
    update()
    map.on("moveend", update)
    map.on("zoomend", update)
    return () => {
      map.off("moveend", update)
      map.off("zoomend", update)
    }
  }, [map])

  if (!isLoaded || !showHeatmap) return null

  const clusterData = buildClusterData(
    stations,
    hour,
    timelineMode,
    getTimelineStationData
  )
  const predMap = buildPredictionLookup(predictions)

  // At low zoom, show the cluster layer only.
  if (zoom <= CLUSTER_MAX_ZOOM) {
    return (
      <MapClusterLayer
        data={clusterData}
        clusterRadius={50}
        clusterMaxZoom={CLUSTER_MAX_ZOOM}
      />
    )
  }

  // At high zoom, render only stations that are within the visible viewport.
  // This keeps the marker count manageable even with hundreds of stations.
  const visibleStations = viewportBbox
    ? stations.filter(
        (s) =>
          s.lat >= viewportBbox.minLat &&
          s.lat <= viewportBbox.maxLat &&
          s.lon >= viewportBbox.minLon &&
          s.lon <= viewportBbox.maxLon
      )
    : stations

  // Hard cap to avoid pathological viewports showing 300+ markers at once.
  const MAX_MARKERS = 120
  const limitedStations = visibleStations.slice(0, MAX_MARKERS)

  return (
    <>
      <MapClusterLayer
        data={clusterData}
        clusterRadius={50}
        clusterMaxZoom={CLUSTER_MAX_ZOOM}
      />
      {limitedStations.map((s) => {
        const pred = predMap[s.id]
        const td = getTimelineStationData?.(s.id)
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
        )
      })}
    </>
  )
}
