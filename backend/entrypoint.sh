#!/bin/bash
set -e

echo "Running migrations..."
alembic upgrade head

echo "Initializing database/seeding..."
python -m backend.seed

echo "Starting server..."
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000
