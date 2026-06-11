#!/bin/bash
set -e

echo "Running migrations..."
alembic upgrade head

echo "Initializing database/seeding (stations & routes)..."
python -m backend.seed

echo "Seeding comprehensive historical data..."
python -m backend.seed_comprehensive

echo "Starting server..."
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000
