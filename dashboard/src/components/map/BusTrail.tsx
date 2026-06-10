import { MapRoute } from "@/components/ui/map";
import { useAnimationStore } from "@/stores/animationStore";

interface BusTrailProps {
  routeColorMap: Record<string, string>;
  maxBuses?: number;
}

/**
 * Renders fading trail polylines behind buses showing their recent path.
 * Each trail is a series of line segments with decreasing opacity.
 */
export default function BusTrail({ routeColorMap, maxBuses = 30 }: BusTrailProps) {
  const trails = useAnimationStore((s) => s.trails);
  const interpolatedPositions = useAnimationStore((s) => s.interpolatedPositions);

  const busIds = Object.keys(trails).slice(0, maxBuses);

  return (
    <>
      {busIds.map((busId) => {
        const trail = trails[busId];
        if (!trail || trail.length < 2) return null;

        const bus = interpolatedPositions[busId];
        if (!bus) return null;

        const color = routeColorMap[bus.route_id] ?? "#888";

        return (
          <MapRoute
            key={`trail-${busId}`}
            id={`trail-${busId}`}
            coordinates={trail}
            color={color}
            width={2}
            opacity={0.25}
          />
        );
      })}
    </>
  );
}