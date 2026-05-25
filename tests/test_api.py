"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestStationsAPI:
    def test_list_stations(self, client):
        response = client.get("/api/v1/stations")
        assert response.status_code == 200
        data = response.json()
        assert "stations" in data
        assert len(data["stations"]) >= 12

    def test_station_forecast(self, client):
        response = client.get("/api/v1/stations/S001/forecast")
        assert response.status_code == 200
        data = response.json()
        assert data["station_id"] == "S001"
        assert len(data["forecast"]) == 24


class TestRoutesAPI:
    def test_list_routes(self, client):
        response = client.get("/api/v1/routes")
        assert response.status_code == 200
        data = response.json()
        assert "routes" in data
        assert len(data["routes"]) >= 5

    def test_route_stops(self, client):
        response = client.get("/api/v1/routes/R1/stops")
        assert response.status_code == 200
        data = response.json()
        assert data["route_id"] == "R1"
        assert "stops" in data


class TestDashboardAPI:
    def test_kpis(self, client):
        response = client.get("/api/v1/dashboard/kpis")
        assert response.status_code == 200
        data = response.json()
        assert "total_stations" in data
        assert "active_routes" in data


class TestAlertsAPI:
    def test_list_alerts(self, client):
        response = client.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert len(data["alerts"]) >= 2

    def test_ack_alert(self, client):
        response = client.post("/api/v1/alerts/1/ack")
        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is True


class TestScenariosAPI:
    def test_run_scenario(self, client):
        response = client.post("/api/v1/scenarios/run", json={
            "name": "Test",
            "modifications": [{"type": "frequency", "target": "R1", "params": {"headway": 5}}],
        })
        assert response.status_code == 200
        data = response.json()
        assert "scenario_id" in data
        assert "changes" in data
