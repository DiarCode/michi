"""Alert service - manages transit alerts with DB-backed storage and auto-generation from thresholds."""
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from backend.models_orm import AlertORM

logger = logging.getLogger(__name__)

ALERT_RULES: list[dict[str, Any]] = [
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


def list_alerts(db: Session, severity: str | None = None, active_only: bool = True) -> list[dict]:
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


def get_alert(db: Session, alert_id: int) -> dict | None:
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


def _has_recent_alert(db: Session, rule_id: str, entity_id: str | None = None, cooldown_minutes: int = 60) -> bool:
    """Check if a recent alert for this rule/entity exists within the cooldown window."""
    cutoff = datetime.now(UTC) - timedelta(minutes=cooldown_minutes)
    query = db.query(AlertORM).filter(AlertORM.what == rule_id, AlertORM.created_at >= cutoff)
    if entity_id:
        query = query.filter((AlertORM.station_id == entity_id) | (AlertORM.route_id == entity_id))
    return query.first() is not None


def generate_auto_alerts(db: Session, stations: list[dict], forecasts: dict[str, list[dict]] | None = None, routes: list[dict] | None = None) -> list[dict]:
    """Auto-generate alerts from threshold rules based on current station and route data.

    Uses a 60-minute cooldown per rule/entity to avoid alert spam.
    """
    new_alerts: list[dict] = []

    # --- Station capacity rules ---
    for station in stations:
        rid = station.get("ridership_24h", 0) or 0
        load_pct = station.get("load_percent", 0) or 0
        if load_pct == 0 and rid > 0:
            load_pct = min(95, int(rid * 0.08 / 30)) if 6 <= datetime.now().hour <= 22 else min(30, int(rid * 0.01 / 30))
        station_id = station.get("id", station.get("stop_id", ""))
        station_name = station.get("name", station_id)

        for rule in ALERT_RULES:
            if rule["metric"] != "station_capacity":
                continue
            if load_pct < rule["threshold"]:
                continue
            if _has_recent_alert(db, rule["id"], station_id):
                continue
            alert = AlertORM(
                severity=rule["severity"],
                title=rule["title_template"].format(station=station_name),
                message=rule["message_template"].format(station=station_name, value=load_pct),
                station_id=station_id,
                created_at=datetime.now(UTC),
                family="auto",
                what=rule["id"],
            )
            db.add(alert)
            db.flush()
            new_alerts.append({
                "id": alert.id, "severity": alert.severity, "title": alert.title,
                "message": alert.message, "station_id": alert.station_id,
                "created_at": alert.created_at.isoformat(), "auto": True, "rule_id": rule["id"],
            })

    # --- Route overload rules ---
    if routes:
        for route in routes:
            route_id = route.get("id", route.get("route_id", ""))
            route_name = route.get("name", route_id)
            avg_ridership = route.get("avg_ridership", 0) or 0
            # Estimate route load from average ridership
            load_pct = min(100, int((avg_ridership / 3000) * 100)) if avg_ridership > 0 else 0
            for rule in ALERT_RULES:
                if rule["metric"] != "route_avg_load":
                    continue
                if load_pct < rule["threshold"]:
                    continue
                if _has_recent_alert(db, rule["id"], route_id):
                    continue
                alert = AlertORM(
                    severity=rule["severity"],
                    title=rule["title_template"].format(route=route_name),
                    message=rule["message_template"].format(route=route_name, value=load_pct),
                    route_id=route_id,
                    created_at=datetime.now(UTC),
                    family="auto",
                    what=rule["id"],
                )
                db.add(alert)
                db.flush()
                new_alerts.append({
                    "id": alert.id, "severity": alert.severity, "title": alert.title,
                    "message": alert.message, "route_id": alert.route_id,
                    "created_at": alert.created_at.isoformat(), "auto": True, "rule_id": rule["id"],
                })

    # --- Forecast spike rules ---
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
                    if _has_recent_alert(db, "forecast_spike", sid):
                        break
                    alert = AlertORM(
                        severity="info",
                        title=f"Forecast spike at {sid}",
                        message=f"Predicted ridership is {int(pct)}% above normal within 2h.",
                        station_id=sid,
                        created_at=datetime.now(UTC),
                        family="auto",
                        what="forecast_spike",
                    )
                    db.add(alert)
                    db.flush()
                    new_alerts.append({
                        "id": alert.id, "severity": alert.severity, "title": alert.title,
                        "message": alert.message, "station_id": alert.station_id,
                        "created_at": alert.created_at.isoformat(), "auto": True, "rule_id": "forecast_spike",
                    })
                    break

    if new_alerts:
        db.commit()
        logger.info("Auto-generated %d alert(s)", len(new_alerts))
    return new_alerts


def get_alert_rules() -> list[dict]:
    return ALERT_RULES


def add_alert_rule(rule: dict[str, Any]) -> dict:
    rule["id"] = rule.get("id", f"rule_{len(ALERT_RULES) + 1}")
    ALERT_RULES.append(rule)
    return rule


def delete_rule(rule_id: str) -> dict | None:
    """Remove a rule by ID. Returns the deleted rule or None if not found."""
    for i, rule in enumerate(ALERT_RULES):
        if rule["id"] == rule_id:
            return ALERT_RULES.pop(i)
    return None
