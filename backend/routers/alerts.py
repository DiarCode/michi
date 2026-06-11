import contextlib

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.models import AlertListResponse
from backend.models_orm import AlertORM, RouteORM
from backend.routers.stations import _get_stations
from backend.services.alert_service import ack_alert as ack
from backend.services.alert_service import (
    add_alert_rule,
    delete_rule,
    generate_auto_alerts,
    get_alert_rules,
    list_alerts,
)
from backend.services.forecast_service import get_forecast

router = APIRouter()


class AlertRuleInput(BaseModel):
    metric: str
    threshold: int
    severity: str
    title_template: str
    message_template: str


@router.get("", response_model=AlertListResponse)
def get_alerts(severity: str | None = None, active_only: bool = True, db: Session = Depends(get_db_session)):
    return {"alerts": list_alerts(db, severity, active_only)}


@router.get("/rich")
def get_rich_alerts(db: Session = Depends(get_db_session)):
    """Return alerts with rich fields (family, what, why, etc.)."""
    alerts = db.query(AlertORM).order_by(AlertORM.created_at.desc()).limit(50).all()
    return {
        "alerts": [
            {
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
            }
            for a in alerts
        ]
    }


@router.get("/active", response_model=AlertListResponse)
def get_active_alerts(db: Session = Depends(get_db_session)):
    return {"alerts": list_alerts(db, active_only=True)}


@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int, db: Session = Depends(get_db_session)):
    return {"acknowledged": ack(db, alert_id), "alert_id": alert_id}


@router.post("/generate")
def trigger_auto_alerts(db: Session = Depends(get_db_session)):
    """Auto-generate alerts from threshold rules using current station and route data."""
    stations = _get_stations(db)
    routes = [{"id": r.id, "name": r.name, "avg_ridership": r.avg_ridership} for r in db.query(RouteORM).all()]
    forecasts = {}
    for s in stations[:12]:
        sid = s.get("id") or s.get("stop_id", "")
        with contextlib.suppress(Exception):
            forecasts[sid] = get_forecast(sid, db=db)
    new_alerts = generate_auto_alerts(db, stations, forecasts if forecasts else None, routes=routes)
    return {"generated": len(new_alerts), "alerts": new_alerts}


@router.get("/rules")
def list_rules():
    return {"rules": get_alert_rules()}


@router.post("/rules")
def create_rule(rule: AlertRuleInput):
    return add_alert_rule(rule.model_dump())


@router.delete("/rules/{rule_id}")
def remove_rule(rule_id: str):
    """Delete an alert rule by ID."""
    deleted = delete_rule(rule_id)
    if not deleted:
        from backend.exceptions import NotFoundException

        raise NotFoundException("AlertRule", rule_id)
    return {"deleted": rule_id}
