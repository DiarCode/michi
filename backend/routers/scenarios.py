from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ScenarioResult
from backend.services.scenario_service import run_scenario

router = APIRouter()


class ScenarioConfig(BaseModel):
    name: str = "Unnamed"
    add_buses: int = Field(default=0, ge=0, description="Additional buses on specified routes")
    remove_buses: int = Field(default=0, ge=0, description="Buses removed from specified routes")
    weather_factor: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Weather multiplier: 0.5=bad, 1.0=normal, 1.5=event day. None=auto-detect from live weather.",
    )
    closed_stations: list[str] = Field(default_factory=list, description="Station IDs to close")
    horizon: int = Field(default=24, ge=1, le=72, description="Hours to forecast ahead")


@router.post("/run", response_model=ScenarioResult)
def run_scenario_endpoint(config: ScenarioConfig, db: Session = Depends(get_db)):
    return run_scenario(config.model_dump(), db=db)
