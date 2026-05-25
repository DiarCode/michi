from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()

MOCK_ALERTS = [
    {"id": 1, "severity": "high", "title": "High demand at Bayterek", "message": "Ridership 35% above forecast during 08:00-09:00.", "station_id": "S003", "created_at": datetime.now(timezone.utc).isoformat()},
    {"id": 2, "severity": "medium", "title": "Route 25 delay", "message": "Average delay 8 minutes due to roadworks on Turan Ave.", "route_id": "R3", "created_at": datetime.now(timezone.utc).isoformat()},
]

@router.get("")
def list_alerts():
    return {"alerts": MOCK_ALERTS}

@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int):
    return {"acknowledged": True, "alert_id": alert_id}
