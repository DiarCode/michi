"""Real-time service - manages bus positions and streaming."""
import json
import random
from pathlib import Path
from typing import Dict, List

# Load real station data for bus simulation
_SEED_PATH = Path(__file__).parent.parent / "data" / "cache" / "astana_network_seed.json"
_REAL_STOPS: List[Dict] = []
_ROUTE_IDS: List[str] = []

try:
    if not _SEED_PATH.exists():
        # Try the root-relative path as well
        _SEED_PATH = Path(__file__).parent.parent.parent / "data" / "cache" / "astana_network_seed.json"

    with open(_SEED_PATH, "r", encoding="utf-8") as f:
        _seed = json.load(f)
        _REAL_STOPS = _seed["stations"]
        _ROUTE_IDS = [r["route_id"] for r in _seed["routes"]]
except (FileNotFoundError, KeyError, Exception):
    pass

_FALLBACK_STOPS = [
    {"name": "Khan Shatyr", "lat": 51.1334, "lon": 71.4244},
    {"name": "Bayterek", "lat": 51.1283, "lon": 71.4305},
    {"name": "Mega Silk Way", "lat": 51.0891, "lon": 71.4050},
    {"name": "Nurzhol Blvd", "lat": 51.1605, "lon": 71.4704},
    {"name": "Astana Arena", "lat": 51.1081, "lon": 71.4024},
    {"name": "Presidential Park", "lat": 51.1250, "lon": 71.4650},
    {"name": "Central Park", "lat": 51.1400, "lon": 71.4550},
    {"name": "Talan Towers", "lat": 51.1280, "lon": 71.4350},
]
_FALLBACK_ROUTE_IDS = ["R12", "R18", "R25", "R31", "R40"]

_STOPS = _REAL_STOPS if _REAL_STOPS else _FALLBACK_STOPS
_ROUTE_IDS = _ROUTE_IDS if _ROUTE_IDS else _FALLBACK_ROUTE_IDS

BUS_POOL = [
    {"bus_id": f"BUS-{i:03d}", "route_id": _ROUTE_IDS[i % len(_ROUTE_IDS)],
     "lat": float(_STOPS[i % len(_STOPS)]["lat"]) + random.uniform(-0.005, 0.005),
     "lon": float(_STOPS[i % len(_STOPS)]["lon"]) + random.uniform(-0.005, 0.005)}
    for i in range(1, 9)
]

_STOP_NAMES = [s["name"] for s in _STOPS]


def get_current_positions() -> List[Dict]:
    for bus in BUS_POOL:
        bus["lat"] += random.uniform(-0.001, 0.001)
        bus["lon"] += random.uniform(-0.001, 0.001)
        bus["speed_kmh"] = random.randint(15, 55)
        bus["occupancy_percent"] = random.randint(20, 95)
        bus["next_stop"] = random.choice(_STOP_NAMES)
        bus["eta_seconds"] = random.randint(30, 300)
    return BUS_POOL
