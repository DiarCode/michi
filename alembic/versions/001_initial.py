"""Initial schema — stations, routes, alerts, ridership, forecasts

Revision ID: 001
Revises: None
Create Date: 2025-05-25
"""
from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('stations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('stop_id', sa.String(20), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('lat', sa.Float(), nullable=False),
        sa.Column('lon', sa.Float(), nullable=False),
        sa.Column('district', sa.String(100), nullable=True),
        sa.Column('ridership_24h', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stop_id'),
    )
    op.create_index('ix_stations_stop_id', 'stations', ['stop_id'])

    op.create_table('routes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('route_id', sa.String(20), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('color', sa.String(10), nullable=True),
        sa.Column('stop_count', sa.Integer(), nullable=True),
        sa.Column('avg_ridership', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('route_id'),
    )
    op.create_index('ix_routes_route_id', 'routes', ['route_id'])

    op.create_table('route_stops',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('route_id', sa.String(20), nullable=False),
        sa.Column('station_id', sa.String(20), nullable=False),
        sa.Column('stop_order', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['route_id'], ['routes.route_id']),
        sa.ForeignKeyConstraint(['station_id'], ['stations.stop_id']),
    )

    op.create_table('alerts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('title', sa.String(300), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('station_id', sa.String(20), nullable=True),
        sa.Column('route_id', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['station_id'], ['stations.stop_id']),
        sa.ForeignKeyConstraint(['route_id'], ['routes.route_id']),
    )

    op.create_table('ridership',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('station_id', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('passengers', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['station_id'], ['stations.stop_id']),
    )
    op.create_index('ix_ridership_timestamp', 'ridership', ['timestamp'])

    op.create_table('forecasts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('station_id', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('predicted', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['station_id'], ['stations.stop_id']),
    )
    op.create_index('ix_forecasts_timestamp', 'forecasts', ['timestamp'])


def downgrade() -> None:
    op.drop_table('forecasts')
    op.drop_table('ridership')
    op.drop_table('alerts')
    op.drop_table('route_stops')
    op.drop_table('routes')
    op.drop_table('stations')
