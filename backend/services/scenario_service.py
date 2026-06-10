"""Scenario service - runs what-if simulations."""
import hashlib
from typing import Any


def run_scenario(config: dict[str, Any]) -> dict[str, Any]:
    """Run a what-if scenario and return comparison metrics."""
    name = config.get("name", "Unnamed")
    modifications = config.get("modifications", [])
    base_ridership = 10000.0
    base_wait = 5.2
    scenario_ridership = base_ridership
    scenario_wait = base_wait
    for mod in modifications:
        mod_type = mod.get("type")
        params = mod.get("params", {})
        if mod_type == "frequency":
            headway = max(1, min(60, params.get("headway", 10)))
            multiplier = max(0.1, 1 + (10 - headway) * 0.02)
            scenario_ridership *= multiplier
            scenario_wait *= max(0.5, headway / 10)
        elif mod_type == "route_add":
            scenario_ridership *= 1.05
        elif mod_type == "station_close":
            scenario_ridership *= 0.92
            scenario_wait *= 1.15
    scenario_ridership = round(scenario_ridership)
    scenario_wait = round(scenario_wait, 1)
    sid = int(hashlib.md5(name.encode()).hexdigest()[:4], 16)
    return {
        "scenario_id": f"scen-{sid:04d}",
        "base_metrics": {"ridership": int(base_ridership), "avg_wait": base_wait},
        "scenario_metrics": {"ridership": int(scenario_ridership), "avg_wait": scenario_wait},
        "changes": {
            "ridership": round((scenario_ridership - base_ridership) / base_ridership * 100, 1),
            "avg_wait": round((scenario_wait - base_wait) / base_wait * 100, 1),
        },
    }
