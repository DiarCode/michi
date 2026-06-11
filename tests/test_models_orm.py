"""Unit tests for SQLAlchemy ORM models."""

from datetime import UTC, datetime

from backend.models_orm import AlertORM, ForecastORM, RidershipORM, RouteORM, RouteStopORM, StationORM


class TestStationORM:
    def test_create_station(self, db_session):
        s = StationORM(stop_id="S999", name="Test", lat=51.0, lon=71.0, district="Esil", ridership_24h=500)
        db_session.add(s)
        db_session.commit()
        result = db_session.query(StationORM).filter_by(stop_id="S999").first()
        assert result is not None
        assert result.name == "Test"
        assert result.ridership_24h == 500

    def test_station_defaults(self, db_session):
        s = StationORM(stop_id="S998", name="Default", lat=51.0, lon=71.0)
        db_session.add(s)
        db_session.commit()
        result = db_session.query(StationORM).filter_by(stop_id="S998").first()
        assert result.district is None
        assert result.ridership_24h == 0


class TestRouteORM:
    def test_create_route(self, db_session):
        r = RouteORM(route_id="R99", name="Test Route", color="#ABCDEF", stop_count=5, avg_ridership=1200.0)
        db_session.add(r)
        db_session.commit()
        result = db_session.query(RouteORM).filter_by(route_id="R99").first()
        assert result.name == "Test Route"


class TestRouteStopORM:
    def test_route_stop_order(self, seeded_session):
        stops = seeded_session.query(RouteStopORM).filter_by(route_id="R1").order_by(RouteStopORM.stop_order).all()
        assert len(stops) == 2
        assert stops[0].station_id == "S001"
        assert stops[1].station_id == "S002"


class TestAlertORM:
    def test_create_alert(self, db_session):
        a = AlertORM(
            severity="high", title="Test Alert", message="msg", station_id="S001", created_at=datetime.now(UTC)
        )
        db_session.add(a)
        db_session.commit()
        assert db_session.query(AlertORM).count() == 1


class TestRidershipORM:
    def test_create_ridership(self, db_session):
        r = RidershipORM(station_id="S001", timestamp=datetime.now(UTC), passengers=150)
        db_session.add(r)
        db_session.commit()
        assert db_session.query(RidershipORM).count() == 1


class TestForecastORM:
    def test_create_forecast(self, db_session):
        f = ForecastORM(
            station_id="S001",
            timestamp=datetime.now(UTC),
            predicted=1200.5,
            confidence=0.92,
            model_version="v1",
        )
        db_session.add(f)
        db_session.commit()
        result = db_session.query(ForecastORM).first()
        assert result.predicted == 1200.5
