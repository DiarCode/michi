import type { Station } from "@/types";

interface Props { stops: Station[]; color?: string }

export default function RoutePolyline({ stops, color = "#2E86AB" }: Props) {
  if (stops.length < 2) return null;
  const points = stops.map((s) => `${((s.lon - 71.25) / 0.4) * 100},${((51.25 - s.lat) / 0.3) * 100}`).join(" ");
  return <polyline points={points} fill="none" stroke={color} strokeWidth={2} opacity={0.6} />;
};
