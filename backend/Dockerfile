FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY tests/ ./tests/
COPY alembic/ ./alembic/
COPY alembic.ini .
# Copy seed data into the backend package for persistence regardless of volumes
COPY data/cache/astana_network_seed.json ./backend/data/cache/astana_network_seed.json
COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh
EXPOSE 8000
CMD ["sh", "./entrypoint.sh"]