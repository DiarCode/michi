FROM python:3.13-slim
WORKDIR /app

# Install system deps for numpy/torch
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
# Install CPU-only PyTorch first to avoid pulling ~2GB CUDA libs
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Copy source code
COPY main.py ./
COPY data/ ./data/
COPY backend/ ./backend/
COPY alembic/ ./alembic/
COPY artifacts/ ./artifacts/
COPY tests/ ./tests/
COPY README.md MODEL_ARCHITECTURE.md ./

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]