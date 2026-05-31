"""Add missing tables — historical_ridership, weather_readings, events,
interventions, model_artifacts, prediction_accuracy

Revision ID: 002
Revises: 001
Create Date: 2025-05-31
"""
from alembic import op
import sqlalchemy as sa

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- historical_ridership ---
    op.create_table('historical_ridership',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('station_id', sa.String(20), nullable=False),
        sa.Column('route_id', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('passengers_boarding', sa.Integer(), nullable=False),
        sa.Column('passengers_alighting', sa.Integer(), nullable=False),
        sa.Column('load', sa.Integer(), nullable=False),
        sa.Column('weather_code', sa.String(10), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('is_holiday', sa.Boolean(), nullable=True),
        sa.Column('is_event_day', sa.Boolean(), nullable=True),
        sa.Column('day_of_week', sa.Integer(), nullable=True),
        sa.Column('hour', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['station_id'], ['stations.stop_id']),
        sa.ForeignKeyConstraint(['route_id'], ['routes.route_id']),
    )
    op.create_index('ix_historical_ridership_station_id', 'historical_ridership', ['station_id'])
    op.create_index('ix_historical_ridership_timestamp', 'historical_ridership', ['timestamp'])

    # --- weather_readings ---
    op.create_table('weather_readings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('precipitation', sa.Float(), nullable=True),
        sa.Column('wind_speed', sa.Float(), nullable=True),
        sa.Column('visibility', sa.Float(), nullable=True),
        sa.Column('weather_code', sa.String(10), nullable=True),
        sa.Column('sudden_change', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_weather_readings_timestamp', 'weather_readings', ['timestamp'])

    # --- events ---
    op.create_table('events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(300), nullable=False),
        sa.Column('venue', sa.String(200), nullable=True),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=False),
        sa.Column('expected_attendance', sa.Integer(), nullable=True),
        sa.Column('affected_routes', sa.Text(), nullable=True),
        sa.Column('affected_stations', sa.Text(), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    # --- interventions ---
    op.create_table('interventions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('alert_id', sa.Integer(), nullable=True),
        sa.Column('intervention_type', sa.String(50), nullable=False),
        sa.Column('route_id', sa.String(20), nullable=True),
        sa.Column('station_id', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('operator_note', sa.Text(), nullable=True),
        sa.Column('predicted_impact', sa.Text(), nullable=True),
        sa.Column('actual_impact', sa.Text(), nullable=True),
        sa.Column('approved_by', sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['alert_id'], ['alerts.id']),
        sa.ForeignKeyConstraint(['route_id'], ['routes.route_id']),
        sa.ForeignKeyConstraint(['station_id'], ['stations.stop_id']),
    )

    # --- model_artifacts ---
    op.create_table('model_artifacts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('artifact_path', sa.String(500), nullable=False),
        sa.Column('metrics_json', sa.Text(), nullable=True),
        sa.Column('training_config_json', sa.Text(), nullable=True),
        sa.Column('dataset_hash', sa.String(64), nullable=True),
        sa.Column('feature_version', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('is_production', sa.Boolean(), nullable=True),
        sa.Column('is_shadow', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('version'),
    )

    # --- prediction_accuracy ---
    op.create_table('prediction_accuracy',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('station_id', sa.String(20), nullable=True),
        sa.Column('route_id', sa.String(20), nullable=True),
        sa.Column('forecast_timestamp', sa.DateTime(), nullable=False),
        sa.Column('horizon_minutes', sa.Integer(), nullable=False),
        sa.Column('predicted', sa.Float(), nullable=False),
        sa.Column('actual', sa.Float(), nullable=True),
        sa.Column('absolute_error', sa.Float(), nullable=True),
        sa.Column('mape', sa.Float(), nullable=True),
        sa.Column('evaluated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_version'], ['model_artifacts.version']),
        sa.ForeignKeyConstraint(['station_id'], ['stations.stop_id']),
        sa.ForeignKeyConstraint(['route_id'], ['routes.route_id']),
    )


def downgrade() -> None:
    op.drop_table('prediction_accuracy')
    op.drop_table('model_artifacts')
    op.drop_table('interventions')
    op.drop_table('events')
    op.drop_table('weather_readings')
    op.drop_table('historical_ridership')