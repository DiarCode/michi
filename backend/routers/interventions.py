"""Intervention workflow API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database import get_db
from backend.models_orm import InterventionORM
from backend.services.intervention_service import (
    create_intervention, list_interventions, get_intervention,
    update_intervention_status, simulate_intervention_impact, INTERVENTION_TYPES,
)

router = APIRouter()


@router.get("/")
def list_interventions_api(status: Optional[str] = None, limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    interventions = list_interventions(status=status, limit=limit)
    return {"interventions": [_to_dict(i) for i in interventions]}


@router.post("/")
def create_intervention_api(alert_id: Optional[int] = None, intervention_type: str = ..., route_id: Optional[str] = None, station_id: Optional[str] = None):
    intervention = create_intervention(alert_id, intervention_type, route_id, station_id)
    return _to_dict(intervention)


@router.get("/types")
def get_intervention_types():
    return {"types": INTERVENTION_TYPES}


@router.get("/simulate")
def simulate_api(intervention_type: str, route_id: Optional[str] = None, station_id: Optional[str] = None):
    return simulate_intervention_impact(intervention_type, route_id, station_id)


@router.get("/{intervention_id}")
def get_intervention_api(intervention_id: int):
    intervention = get_intervention(intervention_id)
    if not intervention:
        return {"error": "Not found"}
    return _to_dict(intervention)


@router.patch("/{intervention_id}")
def update_status_api(intervention_id: int, status: str, approved_by: Optional[str] = None, operator_note: Optional[str] = None):
    intervention = update_intervention_status(intervention_id, status, approved_by, operator_note)
    if not intervention:
        return {"error": "Not found"}
    return _to_dict(intervention)


def _to_dict(i: InterventionORM) -> dict:
    return {
        "id": i.id,
        "alert_id": i.alert_id,
        "intervention_type": i.intervention_type,
        "route_id": i.route_id,
        "station_id": i.station_id,
        "created_at": i.created_at.isoformat() if i.created_at else None,
        "status": i.status,
        "operator_note": i.operator_note,
        "predicted_impact": i.predicted_impact,
        "actual_impact": i.actual_impact,
        "approved_by": i.approved_by,
    }
