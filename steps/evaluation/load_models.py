"""
steps/evaluation/load_models.py

ZenML step — loads all three trained model checkpoints.

What is a ZenML step?
  A @step is just a Python function decorated with @step.
  ZenML automatically:
    - Tracks its inputs and outputs as artifacts
    - Logs execution time and status to the dashboard
    - Caches the output so re-runs skip this step if nothing changed
    - Makes outputs available to downstream steps

This step loads:
  1. EfficientNet-B4 classifier (Lightning checkpoint)
  2. PatchCore results JSON (already computed by anomalib)
  3. U-Net segmenter (Lightning checkpoint)

Returns typed dataclasses so downstream steps have clear contracts.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import yaml
from zenml import step
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from steps.evaluation.torch_device import resolve_torch_device

logger = get_logger(__name__)


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class ClassifierArtifact:
    """Holds the loaded classifier model and its config."""
    model: nn.Module
    cfg: dict
    checkpoint_path: str
    device: str


@dataclass
class AnomalyArtifact:
    """Holds PatchCore results loaded from JSON."""
    results: dict           # per_category AUROC results
    mean_image_auroc: float
    mean_pixel_auroc: float
    results_json_path: str


@dataclass
class SegmenterArtifact:
    """Holds the loaded segmentation model and its config."""
    model: nn.Module
    cfg: dict
    checkpoint_path: str
    device: str


# ── Helper: resolve checkpoint path ──────────────────────────────────────────

def _resolve_ckpt(local_path: str, s3_bucket: str, s3_key: str) -> str:
    """
    Return local checkpoint path.
    If not found locally, download from S3.
    """
    p = Path(local_path)
    if p.exists():
        return str(p)

    # Try S3 fallback
    import boto3, os
    bucket = os.environ.get("SAGEMAKER_S3_BUCKET", s3_bucket)
    logger.warning(f"Checkpoint not found locally: {p}")
    logger.info(f"Downloading from s3://{bucket}/{s3_key}")

    p.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, s3_key, str(p))
    logger.info(f"Downloaded to {p}")
    return str(p)


# ── Steps ─────────────────────────────────────────────────────────────────────

@step
def load_classifier(
    checkpoint_path: str,
    train_config_path: str = "configs/train_classifier.yaml",
    device: str = "cuda",
) -> ClassifierArtifact:
    """
    Load EfficientNet-B4 classifier from Lightning checkpoint.

    ZenML tracks:
      - checkpoint_path as input artifact
      - ClassifierArtifact as output artifact
      - Execution time and memory usage
    """
    import torch
    from models.classifier import DefectClassifier

    device = resolve_torch_device(device, log_warning=logger.warning)
    logger.info(f"Loading classifier from: {checkpoint_path} on {device}")

    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    model = DefectClassifier.load_from_checkpoint(
        checkpoint_path, cfg=train_cfg, map_location=device
    )
    model = model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Classifier loaded — {total_params:,} parameters")

    return ClassifierArtifact(
        model=model,
        cfg=train_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )


@step
def load_anomaly_results(
    results_json_path: str = "checkpoints/anomaly/results_summary.json",
) -> AnomalyArtifact:
    """
    Load PatchCore evaluation results from the JSON produced by train_anomaly.py.

    PatchCore doesn't have a single checkpoint to load — anomalib saves
    per-category memory banks. We load the aggregated results JSON instead.
    """
    p = Path(results_json_path)

    if not p.exists():
        logger.warning(f"Anomaly results not found at {p}")
        logger.warning("Run train_anomaly.py first or download from S3")
        # Return empty artifact so pipeline can continue partially
        return AnomalyArtifact(
            results={},
            mean_image_auroc=0.0,
            mean_pixel_auroc=0.0,
            results_json_path=str(p),
        )

    with open(p) as f:
        data = json.load(f)

    mean_img  = data.get("mean_image_auroc", 0.0)
    mean_pix  = data.get("mean_pixel_auroc", 0.0)

    logger.info(f"PatchCore results loaded")
    logger.info(f"  Mean image AUROC : {mean_img:.4f}")
    logger.info(f"  Mean pixel AUROC : {mean_pix:.4f}")
    logger.info(f"  Categories       : {len(data.get('per_category', {}))}")

    return AnomalyArtifact(
        results=data.get("per_category", {}),
        mean_image_auroc=mean_img,
        mean_pixel_auroc=mean_pix,
        results_json_path=str(p),
    )


@step
def load_segmenter(
    checkpoint_path: str,
    train_config_path: str = "configs/train_segmentation.yaml",
    device: str = "cuda",
) -> SegmenterArtifact:
    """
    Load U-Net segmentation model from Lightning checkpoint.
    """
    from models.segmentation import DefectSegmenter

    device = resolve_torch_device(device, log_warning=logger.warning)
    logger.info(f"Loading segmenter from: {checkpoint_path} on {device}")

    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    model = DefectSegmenter.load_from_checkpoint(
        checkpoint_path, cfg=train_cfg, map_location=device
    )
    model = model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Segmenter loaded — {total_params:,} parameters")

    return SegmenterArtifact(
        model=model,
        cfg=train_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )
