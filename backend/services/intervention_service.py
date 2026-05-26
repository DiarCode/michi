"""Intervention workflow service — CRUD + status tracking for intervention actions."""
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.database import SessionLocal
from backend.models_orm import InterventionORM, AlertORM


INTERVENTION_TYPES = ["dispatch", "short_turn", "hold", "deadhead", "passenger_info", "route_reinforcement"]
VALID_STATUSES = ["pending", "approved", "executing", "completed", "cancelled"]


def create_intervention(alert_id: Optional[int], intervention_type: str, route_id: Optional[str],
                        station_id: Optional[str], predicted_impact: Optional[Dict] = None) -> InterventionORM:
    if intervention_type not in INTERVENTION_TYPES:
        raise ValueError(f"Invalid intervention type: {intervention_type}")
    session = SessionLocal()
    try:
        intervention = InterventionORM(
            alert_id=alert_id,
            intervention_type=intervention_type,
            route_id=route_id,
            station_id=station_id,
            created_at=datetime.now(timezone.utc),
            status="pending",
            predicted_impact=str(predicted_impact) if predicted_impact else None,
        )
        session.add(intervention)
        session.commit()
        session.refresh(intervention)
        return intervention
    finally:
        session.close()


def list_interventions(status: Optional[str] = None, limit: int = 50) -> List[InterventionORM]:
    session = SessionLocal()
    try:
        q = session.query(InterventionORM)
        if status:
            q = q.filter(InterventionORM.status == status)
        return q.order_by(InterventionORM.created_at.desc()).limit(limit).all()
    finally:
        session.close()


def get_intervention(intervention_id: int) -> Optional[InterventionORM]:
    session = SessionLocal()
    try:
        return session.query(InterventionORM).filter(InterventionORM.id == intervention_id).first()
    finally:
        session.close()


def update_intervention_status(intervention_id: int, status: str, approved_by: Optional[str] = None,
                                operator_note: Optional[str] = None, actual_impact: Optional[Dict] = None) -> Optional[InterventionORM]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    session = SessionLocal()
    try:
        intervention = session.query(InterventionORM).filter(InterventionORM.id == intervention_id).first()
        if not intervention:
            return None
        intervention.status = status
        if approved_by:
            intervention.approved_by = approved_by
        if operator_note:
            intervention.operator_note = operator_note
        if actual_impact:
            intervention.actual_impact = str(actual_impact)
        session.commit()
        session.refresh(intervention)
        return intervention
    finally:
        session.close()


def simulate_intervention_impact(intervention_type: str, route_id: Optional[str],
                                  station_id: Optional[str]) -> Dict:
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
