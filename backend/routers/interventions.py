"""Intervention workflow API endpoints."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.models_orm import InterventionORM
from backend.services.intervention_service import (
    INTERVENTION_TYPES,
    create_intervention,
    get_intervention,
    list_interventions,
    simulate_intervention_impact,
    update_intervention_status,
)

router = APIRouter()


class CreateInterventionRequest(BaseModel):
    alert_id: int | None = None
    intervention_type: str
    route_id: str | None = None
    station_id: str | None = None


@router.get("/")
def list_interventions_api(
    status: str | None = None, limit: int = Query(50, le=200), db: Session = Depends(get_db_session)
):
    interventions = list_interventions(db, status=status, limit=limit)
    return {"interventions": [_to_dict(i) for i in interventions]}


@router.post("/")
def create_intervention_api(
    body: CreateInterventionRequest,
    db: Session = Depends(get_db_session),
):
    intervention = create_intervention(db, body.alert_id, body.intervention_type, body.route_id, body.station_id)
    return _to_dict(intervention)


@router.get("/types")
def get_intervention_types():
    return {"types": INTERVENTION_TYPES}


@router.get("/simulate")
def simulate_api(intervention_type: str, route_id: str | None = None, station_id: str | None = None):
    return simulate_intervention_impact(intervention_type, route_id, station_id)


@router.get("/{intervention_id}")
def get_intervention_api(intervention_id: int, db: Session = Depends(get_db_session)):
    intervention = get_intervention(db, intervention_id)
    if not intervention:
        return {"error": "Not found"}
    return _to_dict(intervention)


@router.patch("/{intervention_id}")
def update_status_api(
    intervention_id: int,
    status: str,
    approved_by: str | None = None,
    operator_note: str | None = None,
    db: Session = Depends(get_db_session),
):
    intervention = update_intervention_status(db, intervention_id, status, approved_by, operator_note)
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
