"""Real-time service - manages bus positions and streaming."""
import json
import random
from pathlib import Path

# Load real station data for bus simulation
_SEED_PATH = Path(__file__).parent.parent / "data" / "cache" / "astana_network_seed.json"
_REAL_STOPS: list[dict] = []
_ROUTE_IDS: list[str] = []

try:
    if not _SEED_PATH.exists():
        # Try the root-relative path as well
        _SEED_PATH = Path(__file__).parent.parent.parent / "data" / "cache" / "astana_network_seed.json"

    with open(_SEED_PATH, encoding="utf-8") as f:
        _seed = json.load(f)
        _REAL_STOPS = _seed["stations"]
        _ROUTE_IDS = [r["route_id"] for r in _seed["routes"]]
except (FileNotFoundError, KeyError, Exception):
    pass

_FALLBACK_STOPS = [
    {"name": "Хан Шатыр", "lat": 51.1335, "lon": 71.4069},
    {"name": "Байтерек", "lat": 51.1300, "lon": 71.4345},
    {"name": "Дом Министерств", "lat": 51.1297, "lon": 71.4381},
    {"name": "ТЦ Keruen City", "lat": 51.1435, "lon": 71.4109},
    {"name": "Астана Опера", "lat": 51.1362, "lon": 71.4085},
    {"name": "Министерство обороны", "lat": 51.1260, "lon": 71.4306},
    {"name": "Парк Ж. Жабаева", "lat": 51.1521, "lon": 71.4433},
    {"name": "Дворец Жастар", "lat": 51.1717, "lon": 71.4275},
]
_FALLBACK_ROUTE_IDS = ["R12", "R18", "R24", "R30", "R36", "R42", "R48", "R54", "R60", "R66"]

_STOPS = _REAL_STOPS if _REAL_STOPS else _FALLBACK_STOPS
_ROUTE_IDS = _ROUTE_IDS if _ROUTE_IDS else _FALLBACK_ROUTE_IDS

# Generate a realistic bus fleet: 3-5 buses per route
BUS_POOL = [
    {"bus_id": f"BUS-{i:03d}", "route_id": _ROUTE_IDS[i % len(_ROUTE_IDS)],
     "lat": float(_STOPS[i % len(_STOPS)]["lat"]) + random.uniform(-0.003, 0.003),
     "lon": float(_STOPS[i % len(_STOPS)]["lon"]) + random.uniform(-0.003, 0.003)}
    for i in range(1, 1 + len(_ROUTE_IDS) * 4)  # ~4 buses per route
]

_STOP_NAMES = [s["name"] for s in _STOPS]


def get_current_positions() -> list[dict]:
    for bus in BUS_POOL:
        bus["lat"] += random.uniform(-0.001, 0.001)
        bus["lon"] += random.uniform(-0.001, 0.001)
        bus["speed_kmh"] = random.randint(15, 55)
        bus["occupancy_percent"] = random.randint(20, 95)
        bus["next_stop"] = random.choice(_STOP_NAMES)
        bus["eta_seconds"] = random.randint(30, 300)
    return BUS_POOL
