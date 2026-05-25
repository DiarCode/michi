from fastapi import APIRouter

router = APIRouter()

@router.get("")
def list_alerts():
    return {"alerts": []}

@router.post("/{alert_id}/ack")
def ack_alert(alert_id: int):
    return {"acknowledged": True}
