"""Unit tests for scenario service."""

from unittest.mock import patch

import numpy as np

from backend.services.scenario_service import _compute_deltas, _heuristic_forecast, run_scenario


class TestHeuristicForecast:
    def test_basic_heuristic(self):
        stations = [
            {"stop_id": "S001", "ridership_24h": 2400},
            {"stop_id": "S002", "ridership_24h": 1200},
        ]
        result = _heuristic_forecast(stations, horizons=[60])
        assert len(result) == 2
        assert result[0]["station_id"] == "S001"
        assert result[0]["predicted"] >= 0
        assert result[0]["model_version"] == "heuristic"

    def test_weather_factor_reduces_ridership(self):
        np.random.seed(42)
        stations = [{"stop_id": "S001", "ridership_24h": 2400}]
        normal = _heuristic_forecast(stations, horizons=[60], weather_factor=1.0)
        np.random.seed(42)
        bad = _heuristic_forecast(stations, horizons=[60], weather_factor=0.5)
        assert bad[0]["predicted"] < normal[0]["predicted"]

    def test_closed_station_zero_ridership(self):
        stations = [{"stop_id": "S001", "ridership_24h": 2400}]
        result = _heuristic_forecast(stations, horizons=[60], closed_stations=["S001"])
        assert result[0]["predicted"] == 0

    def test_add_buses_increases_ridership(self):
        np.random.seed(42)
        stations = [{"stop_id": "S001", "ridership_24h": 2400}]
        normal = _heuristic_forecast(stations, horizons=[60])
        np.random.seed(42)
        boosted = _heuristic_forecast(stations, horizons=[60], add_buses=5)
        assert boosted[0]["predicted"] > normal[0]["predicted"]

    def test_remove_buses_decreases_ridership(self):
        np.random.seed(42)
        stations = [{"stop_id": "S001", "ridership_24h": 2400}]
        normal = _heuristic_forecast(stations, horizons=[60])
        np.random.seed(42)
        reduced = _heuristic_forecast(stations, horizons=[60], remove_buses=5)
        assert reduced[0]["predicted"] < normal[0]["predicted"]


class TestComputeDeltas:
    def test_basic_deltas(self):
        baseline = [
            {"station_id": "S001", "predicted": 100.0},
            {"station_id": "S002", "predicted": 50.0},
        ]
        perturbed = [
            {"station_id": "S001", "predicted": 120.0},
            {"station_id": "S002", "predicted": 45.0},
        ]
        station_names = {"S001": "Station A", "S002": "Station B"}
        deltas, summary = _compute_deltas(baseline, perturbed, station_names)
        assert len(deltas) == 2
        assert summary["total_ridership_change"] == 15.0  # (120-100) + (45-50) = 15
        assert summary["most_affected_station"] == "S001"
        assert summary["least_affected_station"] == "S002"

    def test_empty_forecasts(self):
        deltas, summary = _compute_deltas([], [], {})
        assert len(deltas) == 0
        assert summary["most_affected_station"] == ""
        assert summary["least_affected_station"] == ""


class TestRunScenarioWithDB:
    def test_scenario_with_seeded_db(self, seeded_session):
        config = {
            "name": "Bad Weather",
            "add_buses": 0,
            "remove_buses": 0,
            "weather_factor": 0.5,
            "closed_stations": [],
            "horizon": 24,
        }
        with patch("backend.services.scenario_service.get_cached_model", return_value=(None, None)):
            result = run_scenario(config, db=seeded_session)
        assert "scenario_id" in result
        assert "baseline_forecasts" in result
        assert "perturbed_forecasts" in result
        assert "deltas" in result
        assert "summary" in result
        assert len(result["baseline_forecasts"]) > 0
        assert len(result["perturbed_forecasts"]) > 0

    def test_closed_station_scenario(self, seeded_session):
        config = {
            "name": "Station Closure",
            "add_buses": 0,
            "remove_buses": 0,
            "weather_factor": 1.0,
            "closed_stations": ["S001"],
            "horizon": 24,
        }
        with patch("backend.services.scenario_service.get_cached_model", return_value=(None, None)):
            result = run_scenario(config, db=seeded_session)
        # Closed station should have zero perturbed ridership
        for entry in result["perturbed_forecasts"]:
            if entry["station_id"] == "S001":
                assert entry["predicted"] == 0

    def test_add_buses_scenario(self, seeded_session):
        config_normal = {
            "name": "Normal",
            "add_buses": 0,
            "remove_buses": 0,
            "weather_factor": 1.0,
            "closed_stations": [],
            "horizon": 24,
        }
        config_boost = {
            "name": "More Buses",
            "add_buses": 10,
            "remove_buses": 0,
            "weather_factor": 1.0,
            "closed_stations": [],
            "horizon": 24,
        }
        with patch("backend.services.scenario_service.get_cached_model", return_value=(None, None)):
            result_normal = run_scenario(config_normal, db=seeded_session)
            result_boost = run_scenario(config_boost, db=seeded_session)
        # More buses should result in higher total ridership
        normal_total = sum(d["baseline"] for d in result_normal["deltas"])
        boost_total = sum(d["perturbed"] for d in result_boost["deltas"])
        assert boost_total >= normal_total
