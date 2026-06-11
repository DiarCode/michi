import type {
  Station,
  BusPosition,
  PredictionPoint,
  TimelineMode,
  TimelinePoint,
} from "@/types"
import { Map, MapControls } from "@/components/ui/map"
import ZoomAwareStations from "./ZoomAwareStations"
import BusMarker from "./BusMarker"
import ConfidenceOverlay from "./ConfidenceOverlay"

interface Props {
  stations: Station[]
  buses: BusPosition[]
  hour?: number
  onStationClick?: (stationId: string) => void
  showHeatmap?: boolean
  predictions?: PredictionPoint[]
  /** Timeline mode — controls marker styling and data source */
  timelineMode?: TimelineMode
  /** Get timeline data for a station at the selected time */
  getTimelineStationData?: (stationId: string) => TimelinePoint | undefined
  /** Station data with confidence intervals for the overlay */
  confidenceStations?: Station[]
  /** Whether the confidence overlay is visible */
  showConfidence?: boolean
}

const ASTANA_CENTER: [number, number] = [71.47, 51.13]
const ASTANA_ZOOM = 11

export default function MapContainer({
  stations,
  buses,
  hour = new Date().getHours(),
  onStationClick,
  showHeatmap = true,
  predictions = [],
  timelineMode,
  getTimelineStationData,
  confidenceStations,
  showConfidence = false,
}: Props) {
  return (
    <div className="relative h-full w-full">
      <Map
        center={ASTANA_CENTER}
        zoom={ASTANA_ZOOM}
        className="h-full w-full overflow-hidden rounded-lg"
      >
        <MapControls showZoom showCompass />

        <ZoomAwareStations
          stations={stations}
          hour={hour}
          predictions={predictions}
          timelineMode={timelineMode}
          onStationClick={onStationClick}
          getTimelineStationData={getTimelineStationData}
          showHeatmap={showHeatmap}
        />

        {showConfidence &&
          confidenceStations &&
          confidenceStations.length > 0 && (
            <ConfidenceOverlay stations={confidenceStations} />
          )}

        {buses.map((b) => (
          <BusMarker key={b.bus_id} bus={b} />
        ))}
      </Map>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 space-y-1 rounded-lg bg-white/90 p-2 text-xs shadow-md backdrop-blur-sm dark:bg-gray-900/90">
        <div className="font-semibold text-gray-700 dark:text-gray-300">
          Load Level
        </div>
        <div className="flex items-center gap-1.5 dark:text-gray-300">
          <span className="h-3 w-3 rounded-full bg-green-500" /> &lt;50%
        </div>
        <div className="flex items-center gap-1.5 dark:text-gray-300">
          <span className="h-3 w-3 rounded-full bg-amber-500" /> 50–80%
        </div>
        <div className="flex items-center gap-1.5 dark:text-gray-300">
          <span className="h-3 w-3 rounded-full bg-red-500" /> &gt;80%
        </div>
        {timelineMode === "historical" && (
          <div className="mt-1 space-y-1 border-t pt-1 dark:border-gray-700">
            <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
              <span className="h-3 w-3 rounded-full border-2 border-gray-400" />{" "}
              Past (actual)
            </div>
            <div className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400">
              <span className="h-3 w-3 rounded-full border-2 border-dashed border-purple-600 bg-purple-400/30" />{" "}
              Future (predicted)
            </div>
          </div>
        )}
        {predictions.length > 0 && (
          <div className="mt-1 border-t pt-1 text-gray-500 dark:border-gray-700 dark:text-gray-400">
            Showing +{predictions[0]?.horizon_minutes ?? 0}m predictions
          </div>
        )}
        {showConfidence && (
          <div className="mt-1 space-y-1 border-t pt-1 dark:border-gray-700">
            <div className="font-semibold text-gray-700 dark:text-gray-300">
              Confidence
            </div>
            <div className="flex items-center gap-1.5 dark:text-gray-300">
              <span className="h-3 w-3 rounded-full bg-green-500 opacity-60" />{" "}
              Narrow (&lt;50)
            </div>
            <div className="flex items-center gap-1.5 dark:text-gray-300">
              <span className="h-3 w-3 rounded-full bg-amber-500 opacity-60" />{" "}
              Medium (50–150)
            </div>
            <div className="flex items-center gap-1.5 dark:text-gray-300">
              <span className="h-3 w-3 rounded-full bg-red-500 opacity-60" />{" "}
              Wide (&gt;150)
            </div>
          </div>
        )}
      </div>

      {/* Info overlay */}
      <div className="absolute top-4 left-4 z-10 rounded-lg bg-white/90 p-3 shadow-md backdrop-blur-sm dark:bg-gray-900/90">
        <h3 className="text-sm font-bold text-gray-800 dark:text-white">
          {timelineMode === "historical" ? "Historical View" : "Live Tracking"}
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {buses.length} buses active · {String(hour).padStart(2, "0")}:00
        </p>
      </div>
    </div>
  )
}
