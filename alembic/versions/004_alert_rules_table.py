"""Add alert_rules table for DB-backed alert rule configuration

Revision ID: 004
Revises: 003
Create Date: 2026-06-10
"""
from alembic import op
import sqlalchemy as sa

revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('alert_rules',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('metric', sa.String(50), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_alert_rules_id', 'alert_rules', ['id'])


def downgrade() -> None:
    op.drop_index('ix_alert_rules_id', table_name='alert_rules')
    op.drop_table('alert_rules')