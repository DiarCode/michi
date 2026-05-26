from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.services.alert_service import list_alerts, ack_alert as ack, generate_auto_alerts, get_alert_rules, add_alert_rule
from backend.services.forecast_service import get_forecast
from backend.routers.stations import _get_stations
from backend.database import get_db
from backend.models import AlertListResponse
from backend.models_orm import AlertORM
from sqlalchemy.orm import Session

router = APIRouter()

class AlertRuleInput(BaseModel):
    metric: str
    threshold: int
    severity: str
    title_template: str
    message_template: str

@router.get("", response_model=AlertListResponse)
def get_alerts(severity: Optional[str] = None, active_only: bool = True):
    return {"alerts": list_alerts(severity, active_only)}

@router.get("/rich")
def get_rich_alerts(db: Session = Depends(get_db)):
    """Return alerts with rich fields (family, what, why, etc.)."""
    alerts = db.query(AlertORM).order_by(AlertORM.created_at.desc()).limit(50).all()
    return {"alerts": [{
        "id": a.id,
        "family": a.family,
        "severity": a.severity,
        "title": a.title,
        "what": a.what,
        "when_hint": a.when_hint,
        "where_hint": a.where_hint,
        "why": a.why,
        "confidence": a.confidence,
        "consequence_if_ignored": a.consequence_if_ignored,
        "sla_timer_minutes": a.sla_timer_minutes,
        "acknowledged": a.acknowledged or False,
        "assigned_to": a.assigned_to,
        "station_id": a.station_id,
        "route_id": a.route_id,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    } for a in alerts]}

@router.get("/active", response_model=AlertListResponse)
def get_active_alerts():
    return {"alerts": list_alerts(active_only=True)}

@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int):
    return {"acknowledged": ack(alert_id), "alert_id": alert_id}

@router.post("/generate")
def trigger_auto_alerts(db: Session = Depends(get_db)):
    """Auto-generate alerts from threshold rules using current station data."""
    stations = _get_stations(db)
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
