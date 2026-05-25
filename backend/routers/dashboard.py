from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from backend.services.forecast_service import get_kpi_metrics
from backend.routers.stations import _get_stations
from backend.database import get_db
from backend.models import KPIResponse, OperationsReportResponse
from sqlalchemy.orm import Session
import io, csv
from datetime import datetime, timezone

router = APIRouter()

@router.get("/kpis", response_model=KPIResponse)
def get_kpis(db: Session = Depends(get_db)):
    return get_kpi_metrics(db)

@router.get("/operations", response_model=OperationsReportResponse)
def get_operations_report(report_format: str = Query("json", alias="format"), db: Session = Depends(get_db)):
    """Daily operations summary report."""
    kpis = get_kpi_metrics(db)
    stations = _get_stations(db)

    peak_hours = ["07:00", "08:00", "09:00", "17:00", "18:00", "19:00"]
    district_summary = {}
    for s in stations:
        d = s.get("district", "Unknown")
        if d not in district_summary:
            district_summary[d] = {"stations": 0, "total_ridership": 0}
        district_summary[d]["stations"] += 1
        district_summary[d]["total_ridership"] += s.get("ridership_24h", 0) or 0

    exceptions = [s for s in stations if (s.get("ridership_24h", 0) or 0) > 3000]
    report = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "kpis": kpis,
        "district_summary": district_summary,
        "peak_hours": peak_hours,
        "over_capacity_stations": [{"id": s["id"], "name": s.get("name", ""), "ridership_24h": s.get("ridership_24h", 0)} for s in exceptions],
        "total_stations": len(stations),
    }

    if report_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Metric", "Value"])
        for k, v in kpis.items():
            writer.writerow([k, v])
        writer.writerow(["date", report["date"]])
        writer.writerow(["total_stations", len(stations)])
        for d, vals in district_summary.items():
            writer.writerow([f"district_{d}_stations", vals["stations"]])
            writer.writerow([f"district_{d}_ridership", vals["total_ridership"]])
        for exc in exceptions:
            writer.writerow([f"exception_{exc['name']}", exc["ridership_24h"]])
        output.seek(0)
        return StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=operations_{report['date']}.csv"})

    return report