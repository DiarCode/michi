"""Optimization suggestions engine — generates actionable recommendations from predictions and alerts."""
from datetime import datetime, timezone
from typing import Dict, List, Optional


def generate_suggestions(
    predictions: List[Dict],
    alerts: List[Dict],
    stations: List[Dict],
    routes: List[Dict],
    current_weather: Optional[Dict] = None,
    active_events: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Generate optimization suggestions based on current predictions and alerts."""
    suggestions = []
    station_map = {s["stop_id"]: s for s in stations}
    route_map = {r["route_id"]: r for r in routes}

    # 1. Overcrowding risk → dispatch suggestion
    for pred in predictions:
        if pred.get("horizon_minutes") != 30:
            continue
        sid = pred["station_id"]
        predicted = pred["predicted"]
        station = station_map.get(sid, {})
        base = station.get("ridership_24h", 1500) / 24
        if base > 0 and predicted > base * 1.5:
            route_ids = _get_station_routes(sid, routes)
            route_names = [route_map.get(rid, {}).get("name", rid) for rid in route_ids]
            suggestions.append({
                "type": "dispatch",
                "priority": "high" if predicted > base * 2 else "medium",
                "title": f"Consider dispatching reserve bus near {station.get('name', sid)}",
                "description": f"Predicted {int(predicted)} passengers in 30 min ({int(predicted/base*100)}% of normal). Routes: {', '.join(route_names)}",
                "station_id": sid,
                "route_ids": route_ids,
                "predicted_impact": {"ridership_change": round((predicted/base - 1)*100, 1), "wait_time_change": -15},
                "action": "dispatch_reserve",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

    # 2. Bunching detection → hold/release suggestion
    for alert in alerts:
        if "bunching" in alert.get("title", "").lower() or "bunching" in alert.get("what", "").lower():
            route_id = alert.get("route_id", "")
            suggestions.append({
                "type": "hold",
                "priority": "medium",
                "title": f"Hold bus at next stop on {route_map.get(route_id, {}).get('name', route_id)}",
                "description": f"Bunching detected — consider holding for {3}-{5} min to restore headway.",
                "route_id": route_id,
                "predicted_impact": {"ridership_change": -5, "wait_time_change": -8},
                "action": "hold_release",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

    # 3. Event dispersal → pre-position suggestion
    if active_events:
        for event in active_events:
            end_time = event.get("end_time")
            if end_time:
                try:
                    et = datetime.fromisoformat(end_time)
                    now = datetime.now(timezone.utc)
                    if 0 < (et - now).total_seconds() < 3600:
                        affected = event.get("affected_routes", [])
                        if isinstance(affected, str):
                            import json
                            affected = json.loads(affected)
                        route_names = [route_map.get(r, {}).get("name", r) for r in affected]
                        suggestions.append({
                            "type": "preposition",
                            "priority": "high",
                            "title": f"Pre-position buses near {event.get('venue', 'venue')} for dispersal",
                            "description": f"{event.get('name', 'Event')} ends soon. Expected {event.get('expected_attendance', 'many')} attendees. Routes: {', '.join(route_names)}",
                            "route_ids": affected,
                            "predicted_impact": {"ridership_change": 200, "wait_time_change": -25},
                            "action": "preposition",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        })
                except (ValueError, TypeError):
                    pass

    # 4. Weather demand shift → frequency adjustment
    if current_weather and current_weather.get("weather_code") in ("snow", "blizzard", "extreme_cold"):
        affected_routes = [r["route_id"] for r in routes[:5]]
        suggestions.append({
            "type": "frequency_increase",
            "priority": "medium",
            "title": f"Increase frequency on high-demand routes due to {current_weather['weather_code']}",
            "description": f"Severe weather ({current_weather.get('temperature', 0):.0f}°C) increases transit demand by ~20-25%. Consider 10% frequency boost.",
            "route_ids": affected_routes,
            "predicted_impact": {"ridership_change": 20, "wait_time_change": -12},
            "action": "adjust_frequency",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    # 5. Low demand → reallocation suggestion
    low_routes = []
    for pred in predictions:
        if pred.get("horizon_minutes") != 60:
            continue
        sid = pred["station_id"]
        station = station_map.get(sid, {})
        base = station.get("ridership_24h", 1500) / 24
        if base > 0 and pred["predicted"] < base * 0.5:
            low_routes.extend(_get_station_routes(sid, routes))

    high_demand_stations = set()
    for pred in predictions:
        if pred.get("horizon_minutes") != 60:
            continue
        sid = pred["station_id"]
        station = station_map.get(sid, {})
        base = station.get("ridership_24h", 1500) / 24
        if base > 0 and pred["predicted"] > base * 1.5:
            high_demand_stations.add(sid)

    if low_routes and high_demand_stations:
        suggestions.append({
            "type": "reallocation",
            "priority": "low",
            "title": "Consider reallocation from low-demand to high-demand routes",
            "description": f"Move 1 bus from underutilized routes to serve high-demand areas near {len(high_demand_stations)} stations.",
            "predicted_impact": {"ridership_change": 10, "wait_time_change": -8},
            "action": "reallocate",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: priority_order.get(s.get("priority", "low"), 2))
    return suggestions[:10]


def _get_station_routes(station_id: str, routes: List[Dict]) -> List[str]:
    """Get route IDs that serve a station."""
    result = []
    for r in routes:
        stops = r.get("stops", [])
        if station_id in stops or any(s.get("station_id") == station_id for s in stops if isinstance(s, dict)):
            result.append(r["route_id"])
    return result[:3]
