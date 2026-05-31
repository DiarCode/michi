"""Add rich fields to alerts table and missing columns to forecasts table

Revision ID: 003
Revises: 002
Create Date: 2025-05-31
"""
from alembic import op
import sqlalchemy as sa

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- alerts: rich alert fields ---
    op.add_column('alerts', sa.Column('family', sa.String(50), nullable=True))
    op.add_column('alerts', sa.Column('what', sa.String(), nullable=True))
    op.add_column('alerts', sa.Column('when_hint', sa.String(200), nullable=True))
    op.add_column('alerts', sa.Column('where_hint', sa.String(200), nullable=True))
    op.add_column('alerts', sa.Column('why', sa.String(), nullable=True))
    op.add_column('alerts', sa.Column('confidence', sa.Float(), nullable=True))
    op.add_column('alerts', sa.Column('consequence_if_ignored', sa.String(), nullable=True))
    op.add_column('alerts', sa.Column('sla_timer_minutes', sa.Integer(), nullable=True))
    op.add_column('alerts', sa.Column('acknowledged', sa.Boolean(), nullable=True, server_default='0'))
    op.add_column('alerts', sa.Column('assigned_to', sa.String(100), nullable=True))

    # --- forecasts: missing columns ---
    op.add_column('forecasts', sa.Column('horizon_minutes', sa.Integer(), nullable=True))
    op.add_column('forecasts', sa.Column('route_id', sa.String(20), nullable=True))


def downgrade() -> None:
    # --- forecasts: drop added columns ---
    op.drop_column('forecasts', 'route_id')
    op.drop_column('forecasts', 'horizon_minutes')

    # --- alerts: drop rich fields ---
    op.drop_column('alerts', 'assigned_to')
    op.drop_column('alerts', 'acknowledged')
    op.drop_column('alerts', 'sla_timer_minutes')
    op.drop_column('alerts', 'consequence_if_ignored')
    op.drop_column('alerts', 'confidence')
    op.drop_column('alerts', 'why')
    op.drop_column('alerts', 'where_hint')
    op.drop_column('alerts', 'when_hint')
    op.drop_column('alerts', 'what')
    op.drop_column('alerts', 'family')