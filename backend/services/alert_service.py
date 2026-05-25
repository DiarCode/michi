"""Alert service - manages transit alerts."""
from datetime import datetime, timezone
from typing import List, Optional

ALERTS = [
    {"id": 1, "severity": "high", "title": "High demand at Bayterek",
     "message": "Ridership 35% above forecast during 08:00-09:00.", "station_id": "S003",
     "created_at": datetime.now(timezone.utc).isoformat()},
    {"id": 2, "severity": "medium", "title": "Route 25 delay",
     "message": "Average delay 8 minutes due to roadworks on Turan Ave.", "route_id": "R3",
     "created_at": datetime.now(timezone.utc).isoformat()},
    {"id": 3, "severity": "low", "title": "Weekend schedule change",
     "message": "Route 40 frequency reduced on Sundays.", "route_id": "R5",
     "created_at": datetime.now(timezone.utc).isoformat()},
]

_acked: set = set()


def list_alerts(severity: Optional[str] = None) -> List[dict]:
    result = ALERTS
    if severity:
        result = [a for a in result if a["severity"] == severity]
    for a in result:
        a["acknowledged"] = a["id"] in _acked
    return result


def ack_alert(alert_id: int) -> bool:
    _acked.add(alert_id)
    return True
