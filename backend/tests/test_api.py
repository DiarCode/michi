"""Integration tests for all backend API endpoints."""
import pytest
from datetime import datetime, timezone
from backend.models_orm import AlertORM


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestStationEndpoints:
    def test_list_stations(self, client, seed_stations):
        resp = client.get("/api/v1/stations")
        assert resp.status_code == 200
        data = resp.json()
        stations = data if isinstance(data, list) else data.get("stations", data.get("data", []))
        assert len(stations) >= 5

    def test_station_has_required_fields(self, client, seed_stations):
        resp = client.get("/api/v1/stations")
        data = resp.json()
        stations = data if isinstance(data, list) else data.get("stations", data.get("data", []))
        if stations:
            s = stations[0]
            for key in ["stop_id", "name", "lat", "lon"]:
                assert key in s or key.replace("stop_id", "id") in s, f"Missing: {key}"


class TestRouteEndpoints:
    def test_list_routes(self, client, seed_routes):
        resp = client.get("/api/v1/routes")
        assert resp.status_code == 200
        data = resp.json()
        routes = data if isinstance(data, list) else data.get("routes", data.get("data", []))
        assert len(routes) >= 2

    def test_route_structure(self, client, seed_routes):
        resp = client.get("/api/v1/routes")
        data = resp.json()
        routes = data if isinstance(data, list) else data.get("routes", data.get("data", []))
        if routes:
            r = routes[0]
            assert "route_id" in r or "id" in r
            assert "name" in r


class TestAlertEndpoints:
    def test_list_alerts(self, client, seed_alerts):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert len(data["alerts"]) >= 3

    def test_rich_alerts(self, client, seed_alerts):
        resp = client.get("/api/v1/alerts/rich")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        if data["alerts"]:
            a = data["alerts"][0]
            assert "severity" in a
            assert "title" in a

    def test_active_alerts(self, client, seed_alerts):
        resp = client.get("/api/v1/alerts/active")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data

    def test_ack_alert(self, client, seed_alerts, db):
        alerts = db.query(AlertORM).all()
        if alerts:
            resp = client.post(f"/api/v1/alerts/{alerts[0].id}/ack")
            assert resp.status_code == 200
            data = resp.json()
            assert "acknowledged" in data

    def test_alert_rules(self, client):
        resp = client.get("/api/v1/alerts/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert "rules" in data

    def test_generate_alerts(self, client, seed_stations, seed_routes):
        resp = client.post("/api/v1/alerts/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert "generated" in data


class TestDashboardEndpoints:
    def test_kpis(self, client, seed_stations, seed_routes):
        resp = client.get("/api/v1/dashboard/kpis")
        assert resp.status_code == 200

    def test_operations(self, client, seed_stations):
        resp = client.get("/api/v1/dashboard/operations")
        assert resp.status_code == 200

    def test_suggestions(self, client, seed_stations):
        resp = client.get("/api/v1/dashboard/suggestions")
        assert resp.status_code == 200


class TestAnalyticsEndpoints:
    def test_predictions(self, client, seed_stations):
        resp = client.get("/api/v1/analytics/predictions")
        assert resp.status_code == 200

    def test_trends(self, client, seed_stations):
        resp = client.get("/api/v1/analytics/trends")
        assert resp.status_code == 200

    def test_summary(self, client, seed_stations):
        resp = client.get("/api/v1/analytics/summary")
        assert resp.status_code == 200

    def test_graph(self, client, seed_stations):
        resp = client.get("/api/v1/analytics/graph")
        assert resp.status_code == 200

    def test_compare(self, client, seed_stations):
        resp = client.get("/api/v1/analytics/compare")
        assert resp.status_code == 200

    def test_status(self, client, seed_stations):
        resp = client.get("/api/v1/analytics/status")
        assert resp.status_code == 200


class TestScenarioEndpoints:
    def test_run_scenario(self, client):
        resp = client.post("/api/v1/scenarios/run", json={
            "scenario_type": "what_if",
            "params": {"route_id": "TR01", "buses_added": 2},
        })
        assert resp.status_code in (200, 201, 422)


class TestInterventionEndpoints:
    def test_list_interventions(self, client):
        resp = client.get("/api/v1/interventions/")
        assert resp.status_code == 200

    def test_intervention_types(self, client):
        resp = client.get("/api/v1/interventions/types")
        assert resp.status_code == 200


class TestExecutiveEndpoints:
    def test_executive_kpis(self, client, seed_stations, seed_routes, seed_alerts):
        resp = client.get("/api/v1/executive/kpis")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_executive_roi(self, client):
        resp = client.get("/api/v1/executive/roi")
        assert resp.status_code == 200


class TestDepotEndpoints:
    def test_depot_status(self, client):
        resp = client.get("/api/v1/depot/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "depots" in data

    def test_depot_recommendations(self, client):
        resp = client.get("/api/v1/depot/D1/recommendations")
        assert resp.status_code in (200, 404)


class TestPassengerEndpoints:
    def test_crowding(self, client, seed_stations):
        resp = client.get("/api/v1/passenger/crowding")
        assert resp.status_code == 200

    def test_service_changes(self, client):
        resp = client.get("/api/v1/passenger/service-changes")
        assert resp.status_code == 200

    def test_messaging_templates(self, client):
        resp = client.get("/api/v1/passenger/messaging-templates")
        assert resp.status_code == 200