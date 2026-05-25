from fastapi import APIRouter
from backend.services.forecast_service import get_kpi_metrics

router = APIRouter()

@router.get("/kpis")
def get_kpis():
    return get_kpi_metrics()
