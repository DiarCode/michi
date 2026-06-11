"""Report generation — PDF and CSV exports for executive dashboard data."""

import csv
import io
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database import get_db_session
from backend.routers.executive import (
    get_executive_kpis,
    get_executive_trends,
    get_financial_summary,
    get_roi_summary,
)

router = APIRouter()


def _gather_report_data(db: Session, period: int) -> dict:
    """Gather all executive data for a given period."""
    kpis = get_executive_kpis(db)
    trends = get_executive_trends(days=period, db=db)
    roi = get_roi_summary(db)
    financial = get_financial_summary(db)
    return {
        "kpis": kpis,
        "trends": trends,
        "roi": roi,
        "financial": financial,
    }


def _make_csv(data: dict, period: int) -> str:
    """Build a CSV string from executive report data."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["Section", "Metric", "Value", "Period"])
    writer.writerow([])

    # KPIs
    kpis = data["kpis"]
    for key, value in kpis.items():
        writer.writerow(["KPIs", key.replace("_", " ").title(), value, f"Last {period} days"])
    writer.writerow([])

    # Financial Summary
    daily = data["financial"]["daily"]
    for key, value in daily.items():
        writer.writerow(["Financial Summary", key.replace("_", " ").title(), value, "Daily"])
    monthly = data["financial"]["monthly_projection"]
    for key, value in monthly.items():
        writer.writerow(["Financial Summary", key.replace("_", " ").title(), value, "Monthly Projection"])
    for key, value in data["financial"]["cost_breakdown"].items():
        writer.writerow(["Financial Summary - Cost Breakdown", key.replace("_", " ").title(), value, "Daily"])
    writer.writerow([])

    # ROI
    roi = data["roi"]
    for key, value in roi.items():
        writer.writerow(["ROI", key.replace("_", " ").title(), value, f"Last {period} days"])
    writer.writerow([])

    # Daily Trends
    writer.writerow(["Daily Trends", "Date", "Ridership", f"Last {period} days"])
    for entry in data["trends"]["trends"]:
        writer.writerow(["Daily Trends", entry["date"], entry["ridership"], ""])

    return buf.getvalue()


def _make_pdf(data: dict, period: int) -> bytes:
    """Build a PDF from executive report data using reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        raise RuntimeError("reportlab is not installed. Install it with: pip install reportlab") from None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Michi Transit Intelligence — Executive Report", styles["Title"]))
    now = datetime.now(UTC)
    period_start = now - timedelta(days=period)
    elements.append(
        Paragraph(
            f"Report period: {period_start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')} ({period} days)",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 12))

    # KPI Summary Table
    elements.append(Paragraph("KPI Summary", styles["Heading2"]))
    kpi_table_data = [["Metric", "Value"]]
    for key, value in data["kpis"].items():
        kpi_table_data.append([key.replace("_", " ").title(), str(value)])
    kpi_table = Table(kpi_table_data, colWidths=[120 * mm, 50 * mm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ]
        )
    )
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # Financial Breakdown Table
    elements.append(Paragraph("Financial Breakdown", styles["Heading2"]))
    fin_table_data = [["Metric", "Value"]]
    daily = data["financial"]["daily"]
    for key, value in daily.items():
        fin_table_data.append([key.replace("_", " ").title(), str(value)])
    for key, value in data["financial"]["cost_breakdown"].items():
        fin_table_data.append([f"Cost: {key.replace('_', ' ').title()}", str(value)])
    fin_table = Table(fin_table_data, colWidths=[120 * mm, 50 * mm])
    fin_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ]
        )
    )
    elements.append(fin_table)
    elements.append(Spacer(1, 12))

    # ROI Metrics
    elements.append(Paragraph("ROI Metrics", styles["Heading2"]))
    roi_table_data = [["Metric", "Value"]]
    for key, value in data["roi"].items():
        roi_table_data.append([key.replace("_", " ").title(), str(value)])
    roi_table = Table(roi_table_data, colWidths=[120 * mm, 50 * mm])
    roi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ]
        )
    )
    elements.append(roi_table)
    elements.append(Spacer(1, 12))

    # Trend Summary (text-based)
    elements.append(Paragraph("Trend Summary", styles["Heading2"]))
    trends_data = data["trends"]
    elements.append(
        Paragraph(
            f"Period: {trends_data.get('period_days', period)} days | "
            f"Average daily ridership: {trends_data.get('avg_daily', 'N/A'):,} | "
            f"Trend direction: {trends_data.get('trend', 'N/A')} | "
            f"Change: {trends_data.get('change_pct', 'N/A')}%",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 6))
    if trends_data.get("trends"):
        trend_table_data = [["Date", "Ridership"]]
        for entry in trends_data["trends"]:
            trend_table_data.append([entry["date"], str(entry["ridership"])])
        trend_table = Table(trend_table_data, colWidths=[60 * mm, 60 * mm])
        trend_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                ]
            )
        )
        elements.append(trend_table)

    doc.build(elements)
    return buf.getvalue()


@router.get("/executive")
def get_executive_report(
    format: str = Query("pdf", regex="^(pdf|csv)$"),
    period: int = Query(30, le=90, ge=1),
    db: Session = Depends(get_db_session),
):
    """Generate an executive report as PDF or CSV."""
    data = _gather_report_data(db, period)

    if format == "csv":
        csv_content = _make_csv(data, period)
        now = datetime.now(UTC).strftime("%Y%m%d")
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="executive-report-{now}.csv"',
            },
        )

    # PDF format
    try:
        pdf_bytes = _make_pdf(data, period)
    except RuntimeError as e:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=501,
            content={"detail": str(e), "status": 501},
        )

    now = datetime.now(UTC).strftime("%Y%m%d")
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="executive-report-{now}.pdf"',
        },
    )
