FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
# Install CPU-only PyTorch first to avoid pulling ~2GB CUDA libs
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY tests/ ./tests/
COPY alembic/ ./alembic/
COPY alembic.ini .
# Seed data directory (seed.py loads from here)
RUN mkdir -p ./backend/data/cache
COPY data/cache/astana_network_seed.json ./backend/data/cache/astana_network_seed.json
COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/stations')" || exit 1
CMD ["sh", "./entrypoint.sh"]