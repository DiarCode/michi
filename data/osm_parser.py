"""Parse Astana bus network from OSM data."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"

DISTRICTS = {
    "Esil": {"west": 71.35, "east": 71.55, "south": 51.10, "north": 51.25},
    "Almaty": {"west": 71.40, "east": 71.55, "south": 51.00, "north": 51.12},
    "Saryarka": {"west": 71.30, "east": 71.50, "south": 50.95, "north": 51.08},
    "Baikonur": {"west": 71.25, "east": 71.45, "south": 51.05, "north": 51.20},
}


def _point_in_bbox(lat: float, lon: float, bbox: Dict[str, float]) -> bool:
    return bbox["west"] <= lon <= bbox["east"] and bbox["south"] <= lat <= bbox["north"]


def assign_district(lat: float, lon: float) -> Optional[str]:
    for name, bbox in DISTRICTS.items():
        if _point_in_bbox(lat, lon, bbox):
            return name
    return None


def parse_astana_bus_network(osm_file: Path) -> Dict:
    """Parse bus routes and stops from OSM PBF file."""
    print(f"Parsing OSM file: {osm_file}")

    try:
        import osmnx as ox
        G = ox.graph_from_xml(str(osm_file), retain_all=True)
        nodes, edges = ox.graph_to_gdfs(G)
    except Exception as e:
        print(f"osmnx parse failed: {e}")
        return {}

    bus_stop_mask = nodes.get("highway", "") == "bus_stop"
    bus_stops = nodes[bus_stop_mask].copy()

    if bus_stops.empty:
        print("No bus stops found in OSM data")
        return {}

    stops = []
    for idx, row in bus_stops.iterrows():
        lat, lon = row["y"], row["x"]
        name = row.get("name", f"Stop {idx}")
        district = assign_district(lat, lon)
        stops.append({
            "stop_id": str(idx),
            "name": name if pd.notna(name) else f"Stop {idx}",
            "lat": float(lat),
            "lon": float(lon),
            "district": district or "Unknown",
        })

    route_mask = edges.get("route", "") == "bus"
    bus_edges = edges[route_mask]

    routes = []
    if not bus_edges.empty:
        route_groups = bus_edges.groupby("ref" if "ref" in bus_edges.columns else "name")
        for route_name, group in route_groups:
            if pd.isna(route_name):
                continue
            route_nodes = set()
            for _, edge in group.iterrows():
                route_nodes.add(edge["u"])
                route_nodes.add(edge["v"])

            route_stops = [s for s in stops if s["stop_id"] in route_nodes]
            route_stops.sort(key=lambda s: s["lat"])

            routes.append({
                "route_id": str(route_name).replace(" ", "_"),
                "name": f"Route {route_name}",
                "color": "#2E86AB",
                "stop_ids": [s["stop_id"] for s in route_stops],
            })

    adjacency = {}
    for route in routes:
        for i in range(len(route["stop_ids"]) - 1):
            a, b = route["stop_ids"][i], route["stop_ids"][i + 1]
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

    result = {
        "stops": stops,
        "routes": routes,
        "adjacency": adjacency,
        "metadata": {
            "source": "osm",
            "bbox": {"west": 71.25, "east": 71.65, "south": 50.95, "north": 51.25},
        },
    }

    print(f"Parsed {len(stops)} stops, {len(routes)} routes")
    return result


def save_parsed_network(network: Dict, output_dir: Path = CACHE_DIR):
    output_dir.mkdir(exist_ok=True)
    for key in ["stops", "routes", "adjacency"]:
        path = output_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(network.get(key, {}), f, indent=2, ensure_ascii=False)
    print(f"Saved parsed network to {output_dir}")


def load_parsed_network(input_dir: Path = CACHE_DIR) -> Optional[Dict]:
    try:
        network = {}
        for key in ["stops", "routes", "adjacency"]:
            path = input_dir / f"{key}.json"
            with open(path, "r", encoding="utf-8") as f:
                network[key] = json.load(f)
        return network
    except FileNotFoundError:
        return None


def get_astana_network(use_cache: bool = True) -> Dict:
    """Get parsed Astana bus network, downloading and parsing if needed."""
    if use_cache:
        cached = load_parsed_network()
        if cached:
            return cached

    from data.download_osm import download_osm
    osm_file = download_osm()
    network = parse_astana_bus_network(osm_file)
    if network:
        save_parsed_network(network)
    return network


if __name__ == "__main__":
    network = get_astana_network(use_cache=False)
    print(f"Network: {len(network.get('stops', []))} stops, {len(network.get('routes', []))} routes")
