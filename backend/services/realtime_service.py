"""Real-time service - manages bus positions and streaming."""
import random
from typing import Dict, List

BUS_POOL = [
    {"bus_id": f"BUS-{i:03d}", "route_id": f"R{(i % 5) + 1}",
     "lat": 51.13 + random.uniform(-0.03, 0.03),
     "lon": 71.43 + random.uniform(-0.03, 0.03)}
    for i in range(1, 9)
]

STOP_NAMES = ["Khan Shatyr", "Bayterek", "Mega Silk Way", "Nurzhol Blvd", "Astana Arena",
              "Presidential Park", "Central Park", "Talan Towers"]


def get_current_positions() -> List[Dict]:
    for bus in BUS_POOL:
        bus["lat"] += random.uniform(-0.001, 0.001)
        bus["lon"] += random.uniform(-0.001, 0.001)
        bus["speed_kmh"] = random.randint(15, 55)
        bus["occupancy_percent"] = random.randint(20, 95)
        bus["next_stop"] = random.choice(STOP_NAMES)
        bus["eta_seconds"] = random.randint(30, 300)
    return BUS_POOL
