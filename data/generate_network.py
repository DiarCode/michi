"""Generate realistic Astana transit network from cached OSM data.

Creates deduplicated stops, clusters them into routes, and outputs
JSON seed data ready for the backend database.
"""
import json
import math
from pathlib import Path
from data.osm_parser import assign_district

CACHE_DIR = Path(__file__).parent / "cache"

# Real Astana bus routes (approximated from public transit maps)
ROUTE_DEFINITIONS = [
    {"ref": "12", "name": "Route 12", "color": "#2E86AB", "corridor": "north_south"},
    {"ref": "18", "name": "Route 18", "color": "#A23B72", "corridor": "east_west"},
    {"ref": "25", "name": "Route 25", "color": "#F18F01", "corridor": "central"},
    {"ref": "31", "name": "Route 31", "color": "#C73E1D", "corridor": "peripheral"},
    {"ref": "40", "name": "Route 40", "color": "#3B7A57", "corridor": "express"},
    {"ref": "7", "name": "Route 7", "color": "#6B4C9A", "corridor": "north_loop"},
    {"ref": "15", "name": "Route 15", "color": "#D4A574", "corridor": "south_cross"},
    {"ref": "22", "name": "Route 22", "color": "#4ECDC4", "corridor": "university"},
    {"ref": "33", "name": "Route 33", "color": "#95E1D3", "corridor": "residential"},
    {"ref": "50", "name": "Route 50", "color": "#FF6B6B", "corridor": "business"},
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(a ** 0.5)


def load_and_process_stops(min_separation_m=80):
    raw_path = CACHE_DIR / "astana_bus_stops.json"
    if not raw_path.exists():
        print(f"Run OSM download first — {raw_path} not found")
        return []

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    named = [s for s in raw if s.get("name") and s["name"] != "unnamed"]

    deduped = []
    for s in sorted(named, key=lambda x: x["lat"]):
        too_close = any(
            haversine(s["lat"], s["lon"], d["lat"], d["lon"]) < min_separation_m
            for d in deduped
        )
        if not too_close:
            s["district"] = assign_district(s["lat"], s["lon"]) or "Unknown"
            deduped.append(s)

    print(f"OSM: {len(raw)} raw -> {len(named)} named -> {len(deduped)} deduped")
    return deduped


def assign_stops_to_routes(stops):
    routes = []
    route_stops = {}

    for rdef in ROUTE_DEFINITIONS:
        route_id = f"R{rdef['ref']}"
        corridor = rdef["corridor"]
        selected = []

        if corridor == "north_south":
            selected = [s for s in stops if 71.44 <= s["lon"] <= 71.49 and 51.08 <= s["lat"] <= 51.19]
        elif corridor == "east_west":
            selected = [s for s in stops if 71.38 <= s["lon"] <= 71.48 and 51.12 <= s["lat"] <= 51.17]
        elif corridor == "central":
            selected = [s for s in stops if 71.40 <= s["lon"] <= 71.47 and 51.08 <= s["lat"] <= 51.14]
        elif corridor == "peripheral":
            selected = [s for s in stops if s["district"] in ("Saryarka", "Unknown") and (s["lon"] <= 71.38 or s["lon"] >= 71.50)]
        elif corridor == "express":
            selected = [s for s in stops if 71.35 <= s["lon"] <= 71.55 and 51.06 <= s["lat"] <= 51.20]
        elif corridor == "north_loop":
            selected = [s for s in stops if s["district"] == "Esil" or (71.43 <= s["lon"] <= 71.50 and 51.14 <= s["lat"])]
        elif corridor == "south_cross":
            selected = [s for s in stops if s["lat"] <= 51.12 and 71.35 <= s["lon"] <= 71.50]
        elif corridor == "university":
            selected = [s for s in stops if 71.39 <= s["lon"] <= 71.44 and 51.08 <= s["lat"] <= 51.15]
        elif corridor == "residential":
            selected = [s for s in stops if s["district"] in ("Almaty", "Baikonur")]
        elif corridor == "business":
            selected = [s for s in stops if 71.45 <= s["lon"] <= 71.50 and 51.10 <= s["lat"] <= 51.16]

        if corridor in ("north_south", "express", "residential"):
            selected.sort(key=lambda s: s["lat"])
        elif corridor in ("east_west", "south_cross", "university"):
            selected.sort(key=lambda s: s["lon"])
        elif corridor == "central":
            selected.sort(key=lambda s: s["lat"] + s["lon"])
        elif corridor == "peripheral":
            selected.sort(key=lambda s: s["lon"])
        elif corridor == "north_loop":
            selected.sort(key=lambda s: -s["lat"] + s["lon"])
        elif corridor == "business":
            selected.sort(key=lambda s: s["lon"])

        if len(selected) > 20:
            step = len(selected) // 18
            selected = selected[::step][:20]
        elif len(selected) < 3:
            continue

        route_stops[route_id] = selected[:18]
        routes.append({
            "route_id": route_id,
            "name": rdef["name"],
            "color": rdef["color"],
            "stop_count": len(route_stops[route_id]),
            "avg_ridership": round(1200 + len(route_stops[route_id]) * 80 + hash(rdef["ref"]) % 500, 0),
        })

    return routes, route_stops


def generate():
    stops = load_and_process_stops()
    if not stops:
        return None

    for i, s in enumerate(stops, 1):
        s["stop_id"] = f"S{i:03d}"
        s["ridership_24h"] = int(800 + (hash(s["name"]) % 2400))

    routes, route_stops = assign_stops_to_routes(stops)

    route_stop_data = []
    for route_id, stop_list in route_stops.items():
        for order, stop in enumerate(stop_list, 1):
            route_stop_data.append((route_id, stop["stop_id"], order))

    seed_stations = [
        {"stop_id": s["stop_id"], "name": s["name"], "lat": round(s["lat"], 6),
         "lon": round(s["lon"], 6), "district": s["district"], "ridership_24h": s["ridership_24h"]}
        for s in stops
    ]

    result = {
        "stations": seed_stations,
        "routes": routes,
        "route_stops": route_stop_data,
        "metadata": {
            "source": "osm",
            "total_stations": len(seed_stations),
            "total_routes": len(routes),
            "total_route_stops": len(route_stop_data),
        },
    }

    out_path = CACHE_DIR / "astana_network_seed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Seed: {len(seed_stations)} stations, {len(routes)} routes, {len(route_stop_data)} route-stops")
    return result


if __name__ == "__main__":
    generate()