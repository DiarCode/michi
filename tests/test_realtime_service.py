"""Unit tests for realtime service."""
from backend.services.realtime_service import get_current_positions


class TestRealtimeService:
    def test_get_positions(self):
        positions = get_current_positions()
        assert len(positions) == 8
        assert "bus_id" in positions[0]
        assert "lat" in positions[0]
        assert "speed_kmh" in positions[0]
        assert "occupancy_percent" in positions[0]
