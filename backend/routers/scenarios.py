from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.services.scenario_service import run_scenario
from backend.models import ScenarioResult

router = APIRouter()

class ScenarioConfig(BaseModel):
    name: str
    modifications: List[Dict[str, Any]]

@router.post("/run", response_model=ScenarioResult)
def run_scenario_endpoint(config: ScenarioConfig):
    return run_scenario(config.model_dump())
