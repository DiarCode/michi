from fastapi import APIRouter

router = APIRouter()

@router.get("")
def list_routes():
    return {"routes": []}

@router.get("/{route_id}/stops")
def get_route_stops(route_id: str):
    return {"route_id": route_id, "stops": []}
