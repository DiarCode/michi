"""Update weather_readings table for Open-Meteo integration

Revision ID: 005
Revises: 004
Create Date: 2026-06-10
"""
from alembic import op
import sqlalchemy as sa

revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old columns and add new ones for Open-Meteo integration
    op.drop_index('ix_weather_readings_timestamp', table_name='weather_readings')

    # Rename existing columns where possible, then add/drop as needed
    with op.batch_alter_table('weather_readings') as batch_op:
        # Drop old columns
        batch_op.drop_column('visibility')
        batch_op.drop_column('sudden_change')

        # Rename temperature -> temperature_c
        batch_op.alter_column('temperature', new_column_name='temperature_c')
        # Rename precipitation -> precipitation_mm
        batch_op.alter_column('precipitation', new_column_name='precipitation_mm')
        # Rename wind_speed -> wind_speed_kmh
        batch_op.alter_column('wind_speed', new_column_name='wind_speed_kmh')

        # Change weather_code from String(10) to Integer
        batch_op.alter_column('weather_code',
                               existing_type=sa.String(10),
                               type_=sa.Integer(),
                               existing_nullable=True)

        # Add new columns
        batch_op.add_column(sa.Column('humidity_pct', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('description', sa.String(200), nullable=True))
        batch_op.add_column(sa.Column('is_forecast', sa.Boolean(), nullable=True, server_default='0'))
        batch_op.add_column(sa.Column('source', sa.String(50), nullable=True, server_default='open-meteo'))

    op.create_index('ix_weather_readings_timestamp', 'weather_readings', ['timestamp'])


def downgrade() -> None:
    op.drop_index('ix_weather_readings_timestamp', table_name='weather_readings')

    with op.batch_alter_table('weather_readings') as batch_op:
        # Drop new columns
        batch_op.drop_column('source')
        batch_op.drop_column('is_forecast')
        batch_op.drop_column('description')
        batch_op.drop_column('humidity_pct')

        # Revert weather_code to String
        batch_op.alter_column('weather_code',
                               existing_type=sa.Integer(),
                               type_=sa.String(10),
                               existing_nullable=True)

        # Rename columns back
        batch_op.alter_column('wind_speed_kmh', new_column_name='wind_speed')
        batch_op.alter_column('precipitation_mm', new_column_name='precipitation')
        batch_op.alter_column('temperature_c', new_column_name='temperature')

        # Re-add old columns
        batch_op.add_column(sa.Column('visibility', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('sudden_change', sa.Boolean(), nullable=True, server_default='0'))

    op.create_index('ix_weather_readings_timestamp', 'weather_readings', ['timestamp'])