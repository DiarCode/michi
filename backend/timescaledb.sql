-- TimescaleDB hypertable creation (for production PostgreSQL with TimescaleDB extension)
-- Run this AFTER Alembic migrations on a PostgreSQL+TimescaleDB instance.

-- Convert ridership and forecasts to hypertables for time-series performance
SELECT create_hypertable('ridership', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('forecasts', 'timestamp', if_not_exists => TRUE);

-- Add retention policies (keep 2 years of data)
SELECT add_retention_policy('ridership', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('forecasts', INTERVAL '2 years', if_not_exists => TRUE);

-- Add continuous aggregates for fast dashboard queries
CREATE MATERIALIZED VIEW ridership_hourly
WITH (timescaledb.continuous) AS
SELECT
    station_id,
    time_bucket('1 hour', timestamp) AS bucket,
    AVG(passengers) AS avg_passengers,
    MAX(passengers) AS max_passengers,
    MIN(passengers) AS min_passengers,
    COUNT(*) AS sample_count
FROM ridership
GROUP BY station_id, bucket;

CREATE MATERIALIZED VIEW ridership_daily
WITH (timescaledb.continuous) AS
SELECT
    station_id,
    time_bucket('1 day', timestamp) AS bucket,
    AVG(passengers) AS avg_passengers,
    MAX(passengers) AS max_passengers,
    SUM(passengers) AS total_passengers
FROM ridership
GROUP BY station_id, bucket;

-- Add refresh policies
SELECT add_continuous_aggregate_policy('ridership_hourly', INTERVAL '1 hour', INTERVAL '1 hour', if_not_exists => TRUE);
SELECT add_continuous_aggregate_policy('ridership_daily', INTERVAL '1 day', INTERVAL '1 day', if_not_exists => TRUE);
