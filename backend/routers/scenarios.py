from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.services.scenario_service import run_scenario

router = APIRouter()

class ScenarioConfig(BaseModel):
    name: str
    modifications: List[Dict[str, Any]]

@router.post("/run")
def run_scenario_endpoint(config: ScenarioConfig):
    return run_scenario(config.dict())
