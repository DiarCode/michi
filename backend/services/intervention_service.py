"""Intervention workflow service — CRUD + status tracking for intervention actions."""
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.models_orm import InterventionORM

INTERVENTION_TYPES = ["dispatch", "short_turn", "hold", "deadhead", "passenger_info", "route_reinforcement"]
VALID_STATUSES = ["pending", "approved", "executing", "completed", "cancelled"]


def create_intervention(db: Session, alert_id: int | None, intervention_type: str, route_id: str | None,
                        station_id: str | None, predicted_impact: dict | None = None) -> InterventionORM:
    if intervention_type not in INTERVENTION_TYPES:
        raise ValueError(f"Invalid intervention type: {intervention_type}")
    intervention = InterventionORM(
        alert_id=alert_id,
        intervention_type=intervention_type,
        route_id=route_id,
        station_id=station_id,
        created_at=datetime.now(UTC),
        status="pending",
        predicted_impact=str(predicted_impact) if predicted_impact else None,
    )
    db.add(intervention)
    db.commit()
    db.refresh(intervention)
    return intervention


def list_interventions(db: Session, status: str | None = None, limit: int = 50) -> list[InterventionORM]:
    q = db.query(InterventionORM)
    if status:
        q = q.filter(InterventionORM.status == status)
    return q.order_by(InterventionORM.created_at.desc()).limit(limit).all()


def get_intervention(db: Session, intervention_id: int) -> InterventionORM | None:
    return db.query(InterventionORM).filter(InterventionORM.id == intervention_id).first()


def update_intervention_status(db: Session, intervention_id: int, status: str, approved_by: str | None = None,
                                operator_note: str | None = None, actual_impact: dict | None = None) -> InterventionORM | None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    intervention = db.query(InterventionORM).filter(InterventionORM.id == intervention_id).first()
    if not intervention:
        return None
    intervention.status = status
    if approved_by:
        intervention.approved_by = approved_by
    if operator_note:
        intervention.operator_note = operator_note
    if actual_impact:
        intervention.actual_impact = str(actual_impact)
    db.commit()
    db.refresh(intervention)
    return intervention


def simulate_intervention_impact(intervention_type: str, route_id: str | None,
                                  station_id: str | None) -> dict:
    """Simulate the predicted impact of an intervention (what-if analysis)."""
    base_impacts = {
        "dispatch": {"ridership_change": 15, "wait_time_change": -20, "cost": "1 reserve bus for 2-4 hours"},
        "short_turn": {"ridership_change": -5, "wait_time_change": -10, "cost": "Minimal — route truncation"},
        "hold": {"ridership_change": -3, "wait_time_change": -8, "cost": "3-5 min delay to held bus"},
        "deadhead": {"ridership_change": 10, "wait_time_change": -15, "cost": "1 bus + driver time"},
        "passenger_info": {"ridership_change": 0, "wait_time_change": -5, "cost": "Minimal — notification only"},
        "route_reinforcement": {"ridership_change": 20, "wait_time_change": -25, "cost": "2+ buses reallocated"},
    }
    return base_impacts.get(intervention_type, {"ridership_change": 0, "wait_time_change": 0, "cost": "Unknown"})
