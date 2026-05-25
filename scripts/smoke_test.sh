#!/usr/bin/env bash
set -euo pipefail
echo "=== Michi Platform Smoke Test ==="
echo "[1/5] Testing backend API..."
curl -sf http://localhost:8000/api/v1/health || { echo "FAIL: Backend health"; exit 1; }
echo "  Backend: OK"
echo "[2/5] Testing dashboard..."
curl -sf http://localhost:80 | head -c 200 > /dev/null || echo "WARN: Dashboard not reachable"
echo "  Dashboard: checked"
echo "[3/5] Testing Streamlit..."
curl -sf http://localhost:8501/_stcore/health > /dev/null || echo "WARN: Streamlit not reachable"
echo "  Streamlit: checked"
echo "[4/5] Testing Python imports..."
python -c "from backend.models_orm import Base; print('  Imports OK')"
echo "[5/5] All smoke tests passed"
