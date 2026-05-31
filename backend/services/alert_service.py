"""Alert service - manages transit alerts with DB-backed storage and auto-generation from thresholds."""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from backend.models_orm import AlertORM

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


def list_alerts(db: Session, severity: Optional[str] = None, active_only: bool = True) -> List[dict]:
    """List alerts from the database, optionally filtered by severity or active status."""
    query = db.query(AlertORM).order_by(AlertORM.created_at.desc())
    if severity:
        query = query.filter(AlertORM.severity == severity)
    if active_only:
        query = query.filter(AlertORM.acknowledged == False)

    alerts = query.all()
    return [
        {
            "id": a.id,
            "severity": a.severity,
            "title": a.title,
            "message": a.message or "",
            "station_id": a.station_id,
            "route_id": a.route_id,
            "created_at": a.created_at.isoformat() if a.created_at else "",
            "acknowledged": a.acknowledged or False,
            "auto": a.family == "auto",
            "rule_id": a.what,
        }
        for a in alerts
    ]


def get_alert(db: Session, alert_id: int) -> Optional[dict]:
    """Get a single alert by ID from the database."""
    alert = db.query(AlertORM).filter(AlertORM.id == alert_id).first()
    if not alert:
        return None
    return {
        "id": alert.id,
        "severity": alert.severity,
        "title": alert.title,
        "message": alert.message or "",
        "station_id": alert.station_id,
        "route_id": alert.route_id,
        "created_at": alert.created_at.isoformat() if alert.created_at else "",
        "acknowledged": alert.acknowledged or False,
        "auto": alert.family == "auto",
        "rule_id": alert.what,
    }


def ack_alert(db: Session, alert_id: int) -> bool:
    """Acknowledge an alert by setting its acknowledged flag in the database."""
    alert = db.query(AlertORM).filter(AlertORM.id == alert_id).first()
    if not alert:
        return False
    alert.acknowledged = True
    db.commit()
    return True


def generate_auto_alerts(db: Session, stations: List[dict], forecasts: Optional[Dict[str, List[dict]]] = None) -> List[dict]:
    """Auto-generate alerts from threshold rules based on current station data."""
    new_alerts = []
    for station in stations:
        rid = station.get("ridership_24h", 0) or 0
        load_pct = station.get("load_percent", 0) or 0
        if load_pct == 0 and rid > 0:
            load_pct = min(95, int(rid * 0.08 / 30)) if 6 <= datetime.now().hour <= 22 else min(30, int(rid * 0.01 / 30))

        for rule in ALERT_RULES:
            if rule["metric"] == "station_capacity" and load_pct >= rule["threshold"]:
                existing = db.query(AlertORM).filter(
                    AlertORM.station_id == station.get("id", station.get("stop_id", "")),
                    AlertORM.severity == rule["severity"],
                ).first()
                if not existing:
                    alert = AlertORM(
                        severity=rule["severity"],
                        title=rule["title_template"].format(station=station.get("name", station.get("id", station.get("stop_id", "")))),
                        message=rule["message_template"].format(station=station.get("name", station.get("id", station.get("stop_id", ""))), value=load_pct),
                        station_id=station.get("id", station.get("stop_id", "")),
                        created_at=datetime.now(timezone.utc),
                        family="auto",
                        what=rule["id"],
                    )
                    db.add(alert)
                    db.flush()
                    new_alerts.append({
                        "id": alert.id,
                        "severity": alert.severity,
                        "title": alert.title,
                        "message": alert.message,
                        "station_id": alert.station_id,
                        "created_at": alert.created_at.isoformat(),
                        "auto": True,
                        "rule_id": rule["id"],
                    })

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
                    existing = db.query(AlertORM).filter(
                        AlertORM.station_id == sid,
                        AlertORM.title.like("Forecast spike at%"),
                    ).first()
                    if not existing:
                        alert = AlertORM(
                            severity="info",
                            title=f"Forecast spike at {sid}",
                            message=f"Predicted ridership is {int(pct)}% above normal within 2h.",
                            station_id=sid,
                            created_at=datetime.now(timezone.utc),
                            family="auto",
                            what="forecast_spike",
                        )
                        db.add(alert)
                        db.flush()
                        new_alerts.append({
                            "id": alert.id,
                            "severity": alert.severity,
                            "title": alert.title,
                            "message": alert.message,
                            "station_id": alert.station_id,
                            "created_at": alert.created_at.isoformat(),
                            "auto": True,
                            "rule_id": "forecast_spike",
                        })
                    break

    if new_alerts:
        db.commit()
    return new_alerts


def get_alert_rules() -> List[dict]:
    return ALERT_RULES


def add_alert_rule(rule: Dict[str, Any]) -> dict:
    rule["id"] = rule.get("id", f"rule_{len(ALERT_RULES) + 1}")
    ALERT_RULES.append(rule)
    return rule