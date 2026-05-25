from fastapi import APIRouter
from backend.services.alert_service import list_alerts, ack_alert as ack

router = APIRouter()

@router.get("")
def get_alerts(severity: str = None):
    return {"alerts": list_alerts(severity)}

@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int):
    return {"acknowledged": ack(alert_id), "alert_id": alert_id}
