import { MapRoute } from "@/components/ui/map";
import type { Route } from "@/types";

interface RoutePathLayerProps {
  routes: Route[];
  routePaths: Record<string, [number, number][]>;
  routeColorMap: Record<string, string>;
  highlightedRouteId: string | null;
}

export default function RoutePathLayer({
  routes,
  routePaths,
  routeColorMap,
  highlightedRouteId,
}: RoutePathLayerProps) {
  return (
    <>
      {routes.map((route) => {
        const path = routePaths[route.id];
        if (!path || path.length < 2) return null;

        const isHighlighted = highlightedRouteId === route.id;
        const isDimmed = highlightedRouteId !== null && !isHighlighted;
        const color = routeColorMap[route.id] ?? route.color ?? "#888";

        return (
          <MapRoute
            key={route.id}
            id={route.id}
            coordinates={path}
            color={color}
            width={isHighlighted ? 4 : isDimmed ? 2 : 3}
            opacity={isDimmed ? 0.15 : isHighlighted ? 0.85 : 0.45}
          />
        );
      })}
    </>
  );
}