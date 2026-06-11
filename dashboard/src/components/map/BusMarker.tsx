import { useMemo } from "react"
import type { BusPosition } from "@/types"
import {
  MapMarker,
  MarkerContent,
  MarkerTooltip,
  MarkerPopup,
} from "@/components/ui/map"
import { LOAD_HIGH, LOAD_MID } from "@/lib/constants"
import busIcon from "@/assets/bus.png"

interface Props {
  bus: BusPosition
}

export default function BusMarker({ bus }: Props) {
  const occ = bus.occupancy_percent ?? 0
  const badgeBg =
    occ > LOAD_HIGH
      ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
      : occ > LOAD_MID
        ? "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
        : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"

  // Calculate rotation based on heading/bearing
  // Use speed_kmh to determine if bus is moving; if stationary, use last known heading
  const bearing = useMemo(() => {
    // If we have a heading from the bus position, use it
    if ("heading" in bus && typeof (bus as Record<string, unknown>).heading === "number") {
      return (bus as Record<string, unknown>).heading as number
    }
    // Default heading: 0 (north)
    return 0
  }, [bus])

  const isMoving = (bus.speed_kmh ?? 0) > 0

  return (
    <MapMarker longitude={bus.lon} latitude={bus.lat}>
      <MarkerContent>
        <div className="relative">
          <img
            src={busIcon}
            alt={`Bus ${bus.bus_id}`}
            className="drop-shadow-lg"
            style={{
              width: 28,
              height: 28,
              transform: `rotate(${bearing}deg)`,
              filter: !isMoving ? "grayscale(40%)" : "none",
            }}
          />
          <span className="absolute -top-4 left-1/2 -translate-x-1/2 rounded bg-white/90 px-1 text-[9px] font-bold whitespace-nowrap shadow-sm dark:bg-gray-800/90 dark:text-gray-200">
            {bus.bus_id}
          </span>
          {!isMoving && (
            <span className="absolute -right-1 -top-1 flex size-2.5 items-center justify-center rounded-full bg-gray-400 ring-1 ring-white dark:ring-gray-900" />
          )}
        </div>
      </MarkerContent>
      <MarkerTooltip>
        <div className="text-xs font-semibold">
          {bus.bus_id} · Route {bus.route_id}
        </div>
      </MarkerTooltip>
      <MarkerPopup>
        <div className="min-w-[160px] p-2">
          <div className="text-sm font-bold">{bus.bus_id}</div>
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Route: {bus.route_id}
          </div>
          <div className="text-xs">
            Speed: {bus.speed_kmh ?? "—"} km/h
          </div>
          <div className="text-xs">
            ETA:{" "}
            {bus.eta_seconds ? Math.round(bus.eta_seconds / 60) + " min" : "—"}
          </div>
          <div
            className={`mt-1 inline-block rounded px-1.5 py-0.5 text-[10px] font-bold ${badgeBg}`}
          >
            {occ}% full
          </div>
        </div>
      </MarkerPopup>
    </MapMarker>
  )
}