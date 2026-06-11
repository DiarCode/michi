"""Model artifact store — save/load/checkpoint PyTorch model versions."""

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from sqlalchemy.orm import Session

from backend.models_orm import ModelArtifactORM

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_artifact(
    db: Session,
    model_state: dict,
    metrics: dict[str, float],
    config: dict,
    dataset_hash: str,
    feature_version: int = 1,
    is_production: bool = False,
    is_shadow: bool = False,
) -> ModelArtifactORM:
    """Save a model artifact to disk and register in DB."""
    version = f"dts-gssf-v{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    path = ARTIFACTS_DIR / f"{version}.pt"
    torch.save({"model_state_dict": model_state, "version": version, "config": config}, str(path))

    artifact = ModelArtifactORM(
        version=version,
        artifact_path=str(path),
        metrics_json=json.dumps(metrics),
        training_config_json=json.dumps(config),
        dataset_hash=dataset_hash,
        feature_version=feature_version,
        created_at=datetime.now(UTC),
        is_production=is_production,
        is_shadow=is_shadow,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def get_production_artifact(db: Session) -> ModelArtifactORM | None:
    """Get the current production model artifact."""
    return (
        db.query(ModelArtifactORM)
        .filter(ModelArtifactORM.is_production.is_(True))
        .order_by(ModelArtifactORM.created_at.desc())
        .first()
    )


def get_shadow_artifact(db: Session) -> ModelArtifactORM | None:
    """Get the current shadow (challenger) model artifact."""
    return (
        db.query(ModelArtifactORM)
        .filter(ModelArtifactORM.is_shadow.is_(True))
        .order_by(ModelArtifactORM.created_at.desc())
        .first()
    )


def promote_shadow_to_production(db: Session, shadow_version: str) -> ModelArtifactORM | None:
    """Promote a shadow model to production, demoting current production."""
    # Demote current production
    current_prod = db.query(ModelArtifactORM).filter(ModelArtifactORM.is_production.is_(True)).all()
    for a in current_prod:
        a.is_production = False
    # Promote shadow
    shadow = db.query(ModelArtifactORM).filter(ModelArtifactORM.version == shadow_version).first()
    if shadow:
        shadow.is_shadow = False
        shadow.is_production = True
    db.commit()
    return shadow


def list_artifacts(db: Session, limit: int = 20) -> list:
    """List recent model artifacts."""
    return db.query(ModelArtifactORM).order_by(ModelArtifactORM.created_at.desc()).limit(limit).all()
