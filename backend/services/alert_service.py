"""Alert service - manages transit alerts with auto-generation from thresholds."""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

ALERTS = [
    {"id": 1, "severity": "high", "title": "High demand at Bayterek",
     "message": "Ridership 35% above forecast during 08:00-09:00.", "station_id": "S003",
     "created_at": datetime.now(timezone.utc).isoformat(), "auto": False},
    {"id": 2, "severity": "medium", "title": "Route 25 delay",
     "message": "Average delay 8 minutes due to roadworks on Turan Ave.", "route_id": "R3",
     "created_at": datetime.now(timezone.utc).isoformat(), "auto": False},
    {"id": 3, "severity": "low", "title": "Weekend schedule change",
     "message": "Route 40 frequency reduced on Sundays.", "route_id": "R5",
     "created_at": datetime.now(timezone.utc).isoformat(), "auto": False},
]

ALERT_RULES: List[Dict[str, Any]] = [
    {"id": "rule_capacity_85", "metric": "station_capacity", "threshold": 85, "severity": "warning",
     "title_template": "Station {station} over capacity",
     "message_template": "Load at {station} is {value}% — above 85% threshold."},
    {"id": "rule_capacity_95", "metric": "station_capacity", "threshold": 95, "severity": "critical",
     "title_template": "Station {station} critically overloaded",
     "message_template": "Load at {station} is {value}% — critical capacity breach."},
    {"id": "route_overload", "metric": "route_avg_load", "threshold": 90, "severity": "warning",
     "title_template": "Route {route} overloaded",
     "message_template": "Average load on {route} is {value}% during rush hour."},
    {"id": "forecast_spike", "metric": "forecast_spike", "threshold": 200, "severity": "info",
     "title_template": "Forecast spike at {station}",
     "message_template": "Predicted ridership at {station} is {value}% above normal within 2h."},
]

_acked: set = set()
_next_id = 100


def list_alerts(severity: Optional[str] = None, active_only: bool = True) -> List[dict]:
    result = ALERTS
    if severity:
        result = [a for a in result if a["severity"] == severity]
    if active_only:
        result = [a for a in result if a["id"] not in _acked]
    for a in result:
        a["acknowledged"] = a["id"] in _acked
    return result


def ack_alert(alert_id: int) -> bool:
    _acked.add(alert_id)
    return True


def generate_auto_alerts(stations: List[dict], forecasts: Optional[Dict[str, List[dict]]] = None) -> List[dict]:
    """Auto-generate alerts from threshold rules based on current station data."""
    global _next_id
    new_alerts = []
    for station in stations:
        rid = station.get("ridership_24h", 0) or 0
        load_pct = station.get("load_percent", 0) or 0
        if load_pct == 0 and rid > 0:
            load_pct = min(95, int(rid * 0.08 / 30)) if 6 <= datetime.now().hour <= 22 else min(30, int(rid * 0.01 / 30))

        for rule in ALERT_RULES:
            if rule["metric"] == "station_capacity" and load_pct >= rule["threshold"]:
                existing = any(a.get("station_id") == station["id"] and a.get("severity") == rule["severity"] for a in ALERTS)
                if not existing:
                    alert = {
                        "id": _next_id,
                        "severity": rule["severity"],
                        "title": rule["title_template"].format(station=station.get("name", station["id"])),
                        "message": rule["message_template"].format(station=station.get("name", station["id"]), value=load_pct),
                        "station_id": station["id"],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "auto": True,
                        "rule_id": rule["id"],
                    }
                    ALERTS.append(alert)
                    new_alerts.append(alert)
                    _next_id += 1

    if forecasts:
        for sid, fc in forecasts.items():
            if len(fc) < 2:
                continue
            baseline = fc[0].get("predicted", 0)
            if baseline <= 0:
                continue
            for f in fc[1:min(12, len(fc))]:
                pct = (f.get("predicted", 0) / baseline) * 100
                if pct >= 200:
                    existing = any(a.get("station_id") == sid and "Forecast spike" in a.get("title", "") for a in ALERTS)
                    if not existing:
                        alert = {
                            "id": _next_id,
                            "severity": "info",
                            "title": f"Forecast spike at {sid}",
                            "message": f"Predicted ridership is {int(pct)}% above normal within 2h.",
                            "station_id": sid,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "auto": True,
                            "rule_id": "forecast_spike",
                        }
                        ALERTS.append(alert)
                        new_alerts.append(alert)
                        _next_id += 1
                    break
    return new_alerts


def get_alert_rules() -> List[dict]:
    return ALERT_RULES


def add_alert_rule(rule: Dict[str, Any]) -> dict:
    rule["id"] = rule.get("id", f"rule_{len(ALERT_RULES) + 1}")
    ALERT_RULES.append(rule)
    return rule
