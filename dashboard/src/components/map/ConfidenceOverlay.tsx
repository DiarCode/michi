import { useEffect, useMemo, useRef } from "react"
import MapLibreGL from "maplibre-gl"
import { useMap } from "@/components/ui/map"
import type { Station } from "@/types"
import type * as GeoJSON from "geojson"

interface ConfidenceOverlayProps {
  stations: Station[]
}

const SOURCE_ID = "confidence-overlay-source"
const LAYER_ID = "confidence-overlay-layer"

export default function ConfidenceOverlay({
  stations,
}: ConfidenceOverlayProps) {
  const { map, isLoaded } = useMap()
  const popupRef = useRef<MapLibreGL.Popup | null>(null)

  const geoJSON = useMemo<GeoJSON.FeatureCollection<GeoJSON.Point>>(
    () => ({
      type: "FeatureCollection",
      features: stations
        .filter(
          (s) =>
            s.confidence_lower !== undefined && s.confidence_upper !== undefined
        )
        .map((s) => {
          const width = s.confidence_upper! - s.confidence_lower!
          return {
            type: "Feature" as const,
            geometry: {
              type: "Point" as const,
              coordinates: [s.lon, s.lat],
            },
            properties: {
              id: s.id,
              name: s.name,
              confidence_lower: s.confidence_lower!,
              confidence_upper: s.confidence_upper!,
              interval_width: width,
            },
          }
        }),
    }),
    [stations]
  )

  // Add source and circle layer on mount
  useEffect(() => {
    if (!isLoaded || !map) return

    map.addSource(SOURCE_ID, {
      type: "geojson",
      data: geoJSON,
    })

    map.addLayer({
      id: LAYER_ID,
      type: "circle",
      source: SOURCE_ID,
      paint: {
        "circle-color": [
          "step",
          ["get", "interval_width"],
          "#22c55e",
          50,
          "#eab308",
          150,
          "#ef4444",
        ],
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["get", "interval_width"],
          0,
          8,
          50,
          12,
          150,
          18,
          300,
          25,
        ],
        "circle-opacity": 0.6,
        "circle-stroke-width": 2,
        "circle-stroke-color": "#fff",
        "circle-stroke-opacity": 0.8,
      },
    })

    return () => {
      try {
        if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID)
        if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID)
      } catch {
        // ignore cleanup errors
      }
      popupRef.current?.remove()
      popupRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoaded, map])

  // Update source data when stations change
  useEffect(() => {
    if (!isLoaded || !map) return
    const source = map.getSource(SOURCE_ID) as
      | MapLibreGL.GeoJSONSource
      | undefined
    if (source) {
      source.setData(geoJSON)
    }
  }, [isLoaded, map, geoJSON])

  // Hover popup showing exact confidence values
  useEffect(() => {
    if (!isLoaded || !map) return

    const handleMouseMove = (e: MapLibreGL.MapLayerMouseEvent) => {
      if (!e.features || e.features.length === 0) return
      const props = e.features[0].properties
      if (!props) return

      const width = props.interval_width as number
      const lower = props.confidence_lower as number
      const upper = props.confidence_upper as number
      const name = props.name as string

      map.getCanvas().style.cursor = "pointer"

      if (!popupRef.current) {
        popupRef.current = new MapLibreGL.Popup({
          offset: 12,
          closeButton: false,
          maxWidth: "200px",
        })
      }

      popupRef.current
        .setLngLat(e.lngLat)
        .setHTML(
          `<div style="font-family:system-ui,sans-serif;font-size:12px;padding:2px 0">` +
            `<div style="font-weight:600;margin-bottom:4px">${name}</div>` +
            `<div>Lower bound: <strong>${Math.round(lower)}</strong> pax</div>` +
            `<div>Upper bound: <strong>${Math.round(upper)}</strong> pax</div>` +
            `<div style="margin-top:4px;font-weight:500">Interval width: <strong>${Math.round(width)}</strong></div>` +
            `</div>`
        )
        .addTo(map)
    }

    const handleMouseLeave = () => {
      popupRef.current?.remove()
      popupRef.current = null
      map.getCanvas().style.cursor = ""
    }

    map.on("mousemove", LAYER_ID, handleMouseMove)
    map.on("mouseleave", LAYER_ID, handleMouseLeave)

    return () => {
      map.off("mousemove", LAYER_ID, handleMouseMove)
      map.off("mouseleave", LAYER_ID, handleMouseLeave)
      popupRef.current?.remove()
      popupRef.current = null
      map.getCanvas().style.cursor = ""
    }
  }, [isLoaded, map])

  return null
}
