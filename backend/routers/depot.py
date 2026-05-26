"""Depot operations API — fleet availability, dispatch recommendations."""
from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter()

# Mock depot data (will be replaced with real fleet tracking)
MOCK_DEPOTS = [
    {"depot_id": "D01", "name": "Esil Depot", "lat": 51.128, "lon": 71.430,
     "total_buses": 45, "available": 32, "maintenance": 8, "charging": 5,
     "routes_served": ["R12", "R18", "R40"]},
    {"depot_id": "D02", "name": "Saryarka Depot", "lat": 51.105, "lon": 71.400,
     "total_buses": 38, "available": 25, "maintenance": 6, "charging": 7,
     "routes_served": ["R25", "R31", "R7"]},
    {"depot_id": "D03", "name": "Almaty Depot", "lat": 51.140, "lon": 71.450,
     "total_buses": 30, "available": 20, "maintenance": 5, "charging": 5,
     "routes_served": ["R15", "R22", "R33", "R50"]},
]


@router.get("/status")
def get_depot_status():
    return {"depots": MOCK_DEPOTS}


@router.get("/{depot_id}/dispatch-recommendations")
def get_dispatch_recommendations(depot_id: str):
    depot = next((d for d in MOCK_DEPOTS if d["depot_id"] == depot_id), None)
    if not depot:
        return {"error": "Depot not found"}

    recommendations = []
    if depot["available"] > 0:
        recommendations.append({
            "type": "dispatch",
            "priority": "high",
            "message": f"Deploy 1 reserve bus from {depot['name']} to handle predicted demand surge",
            "available_buses": depot["available"],
            "routes_served": depot["routes_served"],
        })
    if depot["charging"] > 0:
        recommendations.append({
            "type": "wait_for_charge",
            "priority": "low",
            "message": f"{depot['charging']} buses charging, expected available in 30-45 min",
            "buses_charging": depot["charging"],
        })
    return {"depot_id": depot_id, "recommendations": recommendations}
