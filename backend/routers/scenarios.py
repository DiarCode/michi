from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from backend.models import ScenarioResult
from backend.services.scenario_service import run_scenario

router = APIRouter()

class ScenarioConfig(BaseModel):
    name: str
    modifications: list[dict[str, Any]]

@router.post("/run", response_model=ScenarioResult)
def run_scenario_endpoint(config: ScenarioConfig):
    return run_scenario(config.model_dump())
