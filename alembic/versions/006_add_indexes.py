"""Add indexes on ForeignKey columns and evaluated_at

Revision ID: 006
Revises: 005
Create Date: 2026-06-11
"""
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # route_stops
    op.create_index('ix_route_stops_route_id', 'route_stops', ['route_id'])
    op.create_index('ix_route_stops_station_id', 'route_stops', ['station_id'])
    # alerts
    op.create_index('ix_alerts_station_id', 'alerts', ['station_id'])
    op.create_index('ix_alerts_route_id', 'alerts', ['route_id'])
    # forecasts
    op.create_index('ix_forecasts_station_id', 'forecasts', ['station_id'])
    op.create_index('ix_forecasts_route_id', 'forecasts', ['route_id'])
    # interventions
    op.create_index('ix_interventions_alert_id', 'interventions', ['alert_id'])
    op.create_index('ix_interventions_route_id', 'interventions', ['route_id'])
    op.create_index('ix_interventions_station_id', 'interventions', ['station_id'])
    # prediction_accuracy
    op.create_index('ix_prediction_accuracy_station_id', 'prediction_accuracy', ['station_id'])
    op.create_index('ix_prediction_accuracy_evaluated_at', 'prediction_accuracy', ['evaluated_at'])


def downgrade() -> None:
    op.drop_index('ix_prediction_accuracy_evaluated_at', table_name='prediction_accuracy')
    op.drop_index('ix_prediction_accuracy_station_id', table_name='prediction_accuracy')
    op.drop_index('ix_interventions_station_id', table_name='interventions')
    op.drop_index('ix_interventions_route_id', table_name='interventions')
    op.drop_index('ix_interventions_alert_id', table_name='interventions')
    op.drop_index('ix_forecasts_route_id', table_name='forecasts')
    op.drop_index('ix_forecasts_station_id', table_name='forecasts')
    op.drop_index('ix_alerts_route_id', table_name='alerts')
    op.drop_index('ix_alerts_station_id', table_name='alerts')
    op.drop_index('ix_route_stops_station_id', table_name='route_stops')
    op.drop_index('ix_route_stops_route_id', table_name='route_stops')