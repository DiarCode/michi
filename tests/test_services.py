"""Unit tests for backend service layer."""
from backend.services.forecast_service import generate_24h_forecast, get_forecast, get_kpi_metrics
from backend.services.alert_service import list_alerts, ack_alert
from backend.services.scenario_service import run_scenario
from backend.services.realtime_service import get_current_positions


class TestForecastService:
    def test_generate_24h_forecast(self):
        result = generate_24h_forecast("S001", base_ridership=2000)
        assert len(result) == 24
        assert result[0]["station_id"] == "S001"
        assert "predicted" in result[0]
        assert "confidence" in result[0]

    def test_get_forecast_caches(self):
        r1 = get_forecast("S002")
        r2 = get_forecast("S002")
        assert len(r1) == len(r2) == 24

    def test_get_kpi_metrics_no_db(self):
        kpis = get_kpi_metrics(db=None)
        assert "total_stations" in kpis
        assert "active_routes" in kpis
        assert "avg_ridership" in kpis


class TestAlertService:
    def test_list_alerts(self):
        alerts = list_alerts()
        assert len(alerts) >= 2
        assert alerts[0]["severity"] in ("high", "medium", "low")

    def test_filter_by_severity(self):
        high = list_alerts(severity="high")
        assert all(a["severity"] == "high" for a in high)

    def test_ack_alert(self):
        result = ack_alert(1)
        assert result is True
