"""Unit tests for scenario service."""
from backend.services.scenario_service import run_scenario


class TestScenarioService:
    def test_frequency_scenario(self):
        result = run_scenario({"name": "More Buses", "modifications": [{"type": "frequency", "target": "R1", "params": {"headway": 5}}]})
        assert "scenario_id" in result
        assert "base_metrics" in result
        assert result["changes"]["ridership"] > 0

    def test_route_add_scenario(self):
        result = run_scenario({"name": "New Route", "modifications": [{"type": "route_add", "target": "x", "params": {}}]})
        assert result["changes"]["ridership"] == 5.0

    def test_station_close_scenario(self):
        result = run_scenario({"name": "Closed", "modifications": [{"type": "station_close", "target": "S001", "params": {}}]})
        assert result["changes"]["ridership"] < 0
