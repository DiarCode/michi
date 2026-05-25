from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.services.alert_service import list_alerts, ack_alert as ack, generate_auto_alerts, get_alert_rules, add_alert_rule
from backend.services.forecast_service import get_forecast

router = APIRouter()

class AlertRuleInput(BaseModel):
    metric: str
    threshold: int
    severity: str
    title_template: str
    message_template: str

@router.get("")
def get_alerts(severity: Optional[str] = None, active_only: bool = True):
    return {"alerts": list_alerts(severity, active_only)}

@router.get("/active")
def get_active_alerts():
    return {"alerts": list_alerts(active_only=True)}

@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int):
    return {"acknowledged": ack(alert_id), "alert_id": alert_id}

@router.post("/generate")
def trigger_auto_alerts():
    """Auto-generate alerts from threshold rules using current station data."""
    from backend.routers.stations import list_stations
    stations_data = list_stations()
    stations = stations_data.get("stations", []) if isinstance(stations_data, dict) else []
    forecasts = {}
    for s in stations[:12]:
        sid = s.get("id") or s.get("stop_id", "")
        try:
            forecasts[sid] = get_forecast(sid)
        except Exception:
            pass
    new_alerts = generate_auto_alerts(stations, forecasts if forecasts else None)
    return {"generated": len(new_alerts), "alerts": new_alerts}

@router.get("/rules")
def list_rules():
    return {"rules": get_alert_rules()}

@router.post("/rules")
def create_rule(rule: AlertRuleInput):
    return add_alert_rule(rule.model_dump())
