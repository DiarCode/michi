import { useEffect, useRef, useCallback } from "react"
import { useTimelineStore } from "@/stores/timelineStore"
import { fetchTimeline } from "@/lib/api"
import type { TimelinePoint } from "@/types"

const PREFETCH_RANGE_MS = 2 * 60 * 60 * 1000 // ±2h

export function useTimeline() {
  const mode = useTimelineStore((s) => s.mode)
  const currentTime = useTimelineStore((s) => s.currentTime)
  const isPlaying = useTimelineStore((s) => s.isPlaying)
  const playSpeed = useTimelineStore((s) => s.playSpeed)
  const data = useTimelineStore((s) => s.data)
  const range = useTimelineStore((s) => s.range)

  const scrubTo = useTimelineStore((s) => s.scrubTo)
  const togglePlay = useTimelineStore((s) => s.togglePlay)
  const setSpeed = useTimelineStore((s) => s.setSpeed)
  const enterLiveMode = useTimelineStore((s) => s.enterLiveMode)
  const enterHistoricalMode = useTimelineStore((s) => s.enterHistoricalMode)
  const setData = useTimelineStore((s) => s.setData)
  const tick = useTimelineStore((s) => s.tick)

  const lastFetchKeyRef = useRef("")

  const fetchData = useCallback(
    async (centerMs: number) => {
      const start = new Date(centerMs - PREFETCH_RANGE_MS)
      const end = new Date(centerMs + PREFETCH_RANGE_MS)
      const key = `${start.toISOString()}-${end.toISOString()}`
      if (key === lastFetchKeyRef.current) return
      lastFetchKeyRef.current = key
      try {
        const res = await fetchTimeline({
          start_time: start.toISOString(),
          end_time: end.toISOString(),
          resolution: "15m",
        })
        setData(res.timeline ?? [])
      } catch (e) {
        console.error("Failed to fetch timeline data:", e)
      }
    },
    [setData]
  )

  // Fetch timeline data when time changes significantly in historical mode
  useEffect(() => {
    if (mode === "historical") {
      fetchData(currentTime)
    }
  }, [mode, currentTime, fetchData])

  // Auto-play: advance timeline at selected speed (tick every 1s)
  useEffect(() => {
    if (!isPlaying) return
    const interval = setInterval(() => {
      tick()
    }, 1000)
    return () => clearInterval(interval)
  }, [isPlaying, tick])

  // Keep currentTime synced to wall clock in live mode
  useEffect(() => {
    if (mode !== "live" || isPlaying) return
    const interval = setInterval(() => {
      scrubTo(Date.now())
    }, 30000)
    return () => clearInterval(interval)
  }, [mode, isPlaying, scrubTo])

  // Get data for a specific station at the current timeline time
  const getStationData = useCallback(
    (stationId: string): TimelinePoint | undefined => {
      const stationPoints = data.filter((p) => p.station_id === stationId)
      if (stationPoints.length === 0) return undefined

      let closest = stationPoints[0]
      let closestDist = Math.abs(
        new Date(closest.timestamp).getTime() - currentTime
      )
      for (const p of stationPoints) {
        const dist = Math.abs(new Date(p.timestamp).getTime() - currentTime)
        if (dist < closestDist) {
          closest = p
          closestDist = dist
        }
      }
      // Only return if within 30 minutes of requested time
      if (closestDist > 30 * 60 * 1000) return undefined
      return closest
    },
    [data, currentTime]
  )

  // When user grabs scrubber, enter historical mode
  const handleScrubStart = useCallback(
    (timestampMs: number) => {
      const now = Date.now()
      const rangeStart = now - 24 * 60 * 60 * 1000 // -24h
      const rangeEnd = now + 12 * 60 * 60 * 1000 // +12h
      enterHistoricalMode(rangeStart, rangeEnd)
      scrubTo(timestampMs)
    },
    [enterHistoricalMode, scrubTo]
  )

  return {
    mode,
    currentTime,
    isPlaying,
    playSpeed,
    data,
    range,
    getStationData,
    handleScrubStart,
    enterLiveMode,
    togglePlay,
    setSpeed,
    scrubTo,
  }
}
