from fastapi import APIRouter

router = APIRouter()

@router.get("/kpis")
def get_kpis():
    return {
        "total_stations": 12,
        "active_routes": 5,
        "avg_ridership": 1845.0,
        "alerts_today": 2,
        "on_time_performance": 94.2,
        "peak_hour": "08:00",
    }
