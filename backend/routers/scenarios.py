from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

class ScenarioConfig(BaseModel):
    name: str
    modifications: List[Dict[str, Any]]

class ScenarioResult(BaseModel):
    scenario_id: str
    base_metrics: Dict[str, float]
    scenario_metrics: Dict[str, float]
    changes: Dict[str, float]

@router.post("/run")
def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="scen-001",
        base_metrics={"ridership": 10000, "avg_wait": 5.2},
        scenario_metrics={"ridership": 9500, "avg_wait": 4.8},
        changes={"ridership": -5.0, "avg_wait": -7.7},
    )
