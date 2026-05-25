from fastapi import APIRouter

router = APIRouter()

@router.get("/kpis")
def get_kpis():
    return {
        "total_stations": 0,
        "active_routes": 0,
        "avg_ridership": 0.0,
        "alerts_today": 0,
    }
