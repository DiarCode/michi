"""Model artifact store — save/load/checkpoint PyTorch model versions."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch

from backend.database import SessionLocal
from backend.models_orm import ModelArtifactORM

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_artifact(
    model_state: dict,
    metrics: Dict[str, float],
    config: Dict,
    dataset_hash: str,
    feature_version: int = 1,
    is_production: bool = False,
    is_shadow: bool = False,
) -> ModelArtifactORM:
    """Save a model artifact to disk and register in DB."""
    version = f"dts-gssf-v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    path = ARTIFACTS_DIR / f"{version}.pt"
    torch.save({"model_state_dict": model_state, "version": version, "config": config}, str(path))

    session = SessionLocal()
    try:
        artifact = ModelArtifactORM(
            version=version,
            artifact_path=str(path),
            metrics_json=json.dumps(metrics),
            training_config_json=json.dumps(config),
            dataset_hash=dataset_hash,
            feature_version=feature_version,
            created_at=datetime.now(timezone.utc),
            is_production=is_production,
            is_shadow=is_shadow,
        )
        session.add(artifact)
        session.commit()
        session.refresh(artifact)
        return artifact
    finally:
        session.close()


def get_production_artifact() -> Optional[ModelArtifactORM]:
    """Get the current production model artifact."""
    session = SessionLocal()
    try:
        return (session.query(ModelArtifactORM)
                .filter(ModelArtifactORM.is_production == True)
                .order_by(ModelArtifactORM.created_at.desc())
                .first())
    finally:
        session.close()


def get_shadow_artifact() -> Optional[ModelArtifactORM]:
    """Get the current shadow (challenger) model artifact."""
    session = SessionLocal()
    try:
        return (session.query(ModelArtifactORM)
                .filter(ModelArtifactORM.is_shadow == True)
                .order_by(ModelArtifactORM.created_at.desc())
                .first())
    finally:
        session.close()


def promote_shadow_to_production(shadow_version: str) -> Optional[ModelArtifactORM]:
    """Promote a shadow model to production, demoting current production."""
    session = SessionLocal()
    try:
        # Demote current production
        current_prod = (session.query(ModelArtifactORM)
                       .filter(ModelArtifactORM.is_production == True).all())
        for a in current_prod:
            a.is_production = False
        # Promote shadow
        shadow = (session.query(ModelArtifactORM)
                 .filter(ModelArtifactORM.version == shadow_version).first())
        if shadow:
            shadow.is_shadow = False
            shadow.is_production = True
        session.commit()
        return shadow
    finally:
        session.close()


def list_artifacts(limit: int = 20) -> list:
    """List recent model artifacts."""
    session = SessionLocal()
    try:
        return (session.query(ModelArtifactORM)
                .order_by(ModelArtifactORM.created_at.desc())
                .limit(limit).all())
    finally:
        session.close()
