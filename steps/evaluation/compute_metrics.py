"""
steps/evaluation/compute_metrics.py

ZenML steps — compute evaluation metrics for all three models.

Three steps:
  classifier_metrics_step  → AUROC, F1, precision, recall, confusion matrix
  anomaly_metrics_step     → image AUROC, pixel AUROC, benchmark delta
  segmentation_metrics_step → Dice, IoU, pixel AUROC

Each step returns a plain dict — ZenML serializes it as a JSON artifact
and tracks it in the dashboard with full lineage.
"""

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from steps.evaluation.torch_device import resolve_torch_device

logger = get_logger(__name__)


# ── Classifier metrics ────────────────────────────────────────────────────────

@step(enable_cache=False)
def classifier_metrics_step(
    checkpoint_path: str,
    metadata_csv: str,
    train_config_path: str = "configs/train_classifier.yaml",
    threshold: float = 0.5,
    device: str = "cuda",
) -> Annotated[dict, "classifier_metrics"]:
    """
    Run EfficientNet-B4 on the full test set and compute:
      - Binary AUROC
      - F1, precision, recall at threshold
      - Confusion matrix
      - Per-category AUROC (all 15 MVTec categories)
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score,
        recall_score, confusion_matrix, average_precision_score,
    )
    from models.classifier import DefectClassifier
    from data.transforms import get_classifier_transforms
    from data.dataset import DefectClassificationDataset
    from torch.utils.data import DataLoader
    import pandas as pd

    device = resolve_torch_device(device, log_warning=logger.warning)
    logger.info(f"Computing classifier metrics on {device}")

    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    model = DefectClassifier.load_from_checkpoint(
        checkpoint_path, cfg=train_cfg, map_location=device
    ).to(device).eval()

    transforms = get_classifier_transforms(512)
    data_cfg   = yaml.safe_load(open("configs/data.yaml"))

    # ── Full test set ──────────────────────────────────────────────────────
    ds = DefectClassificationDataset(
        csv_path=metadata_csv, split="test", transform=transforms["test"]
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs_all  = np.concatenate(all_probs)
    labels_all = np.concatenate(all_labels)
    preds_all  = (probs_all >= threshold).astype(int)

    auroc = float(roc_auc_score(labels_all, probs_all)) if len(np.unique(labels_all)) > 1 else 0.0

    results = {
        "auroc":            round(auroc, 4),
        "f1":               round(float(f1_score(labels_all, preds_all, zero_division=0)), 4),
        "precision":        round(float(precision_score(labels_all, preds_all, zero_division=0)), 4),
        "recall":           round(float(recall_score(labels_all, preds_all, zero_division=0)), 4),
        "avg_precision":    round(float(average_precision_score(labels_all, probs_all)), 4),
        "threshold":        threshold,
        "n_samples":        int(len(labels_all)),
        "n_normal":         int((labels_all == 0).sum()),
        "n_defective":      int((labels_all == 1).sum()),
        "confusion_matrix": confusion_matrix(labels_all, preds_all).tolist(),
        "_probs":           probs_all.tolist(),
        "_labels":          labels_all.tolist(),
    }

    # ── Per-category AUROC ─────────────────────────────────────────────────
    categories = data_cfg["dataset"]["categories"]
    per_cat    = {}

    for category in categories:
        ds_cat = DefectClassificationDataset(
            csv_path=metadata_csv, split="test",
            transform=transforms["test"], category=category,
        )
        if len(ds_cat) == 0:
            continue

        loader_cat  = DataLoader(ds_cat, batch_size=32, shuffle=False, num_workers=2)
        probs_c, labels_c = [], []

        with torch.no_grad():
            for images, labels in loader_cat:
                logits = model(images.to(device))
                probs_c.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                labels_c.append(labels.numpy())

        p = np.concatenate(probs_c)
        l = np.concatenate(labels_c)

        if len(np.unique(l)) > 1:
            per_cat[category] = round(float(roc_auc_score(l, p)), 4)
            logger.info(f"  [{category:<16}] AUROC={per_cat[category]:.4f}")

    per_cat["_mean"] = round(float(np.mean(list(per_cat.values()))), 4)
    results["per_category_auroc"] = per_cat

    logger.info(f"Classifier AUROC={results['auroc']:.4f}  F1={results['f1']:.4f}")

    # Log key metrics to ZenML dashboard
    log_artifact_metadata(
        metadata={
            "auroc":     results["auroc"],
            "f1":        results["f1"],
            "precision": results["precision"],
            "recall":    results["recall"],
        }
    )

    return results


# ── Anomaly metrics ───────────────────────────────────────────────────────────

@step(enable_cache=True)
def anomaly_metrics_step(
    results_json_path: str = "checkpoints/anomaly/results_summary.json",
) -> Annotated[dict, "anomaly_metrics"]:
    """
    Load PatchCore results and build benchmark comparison vs published numbers.
    """
    import json

    PUBLISHED = {
        "bottle": 0.9996, "cable": 0.9938, "capsule": 0.9820,
        "carpet": 0.9870, "grid": 0.9827, "hazelnut": 1.0000,
        "leather": 1.0000, "metal_nut": 1.0000, "pill": 0.9661,
        "screw": 0.9750, "tile": 0.9874, "toothbrush": 1.0000,
        "transistor": 1.0000, "wood": 0.9891, "zipper": 0.9985,
    }

    p = Path(results_json_path)
    if not p.exists():
        logger.warning(f"Anomaly results not found: {p}")
        return {"error": "results_summary.json not found", "per_category": {}}

    with open(p) as f:
        data = json.load(f)

    per_cat    = data.get("per_category", {})
    comparison = {}

    for cat, pub_auroc in PUBLISHED.items():
        your_auroc = per_cat.get(cat, {}).get("image_auroc")
        delta      = round(your_auroc - pub_auroc, 4) if your_auroc else None
        comparison[cat] = {
            "your_image_auroc":      your_auroc,
            "published_image_auroc": pub_auroc,
            "delta":                 delta,
            "your_pixel_auroc":      per_cat.get(cat, {}).get("pixel_auroc"),
        }

    your_aurocs = [v["your_image_auroc"] for v in comparison.values() if v["your_image_auroc"]]
    mean_your   = round(float(np.mean(your_aurocs)), 4) if your_aurocs else 0.0
    mean_pub    = round(float(np.mean(list(PUBLISHED.values()))), 4)

    results = {
        "mean_image_auroc":           mean_your,
        "mean_pixel_auroc":           data.get("mean_pixel_auroc", 0.0),
        "published_mean_image_auroc": mean_pub,
        "mean_delta":                 round(mean_your - mean_pub, 4),
        "per_category":               comparison,
    }

    logger.info(f"PatchCore mean AUROC     : {mean_your:.4f}")
    logger.info(f"Published mean AUROC     : {mean_pub:.4f}")
    logger.info(f"Delta vs published       : {results['mean_delta']:+.4f}")

    log_artifact_metadata(
        metadata={
            "mean_image_auroc":  mean_your,
            "mean_pixel_auroc":  results["mean_pixel_auroc"],
            "delta_vs_published": results["mean_delta"],
        }
    )

    return results


# ── Segmentation metrics ──────────────────────────────────────────────────────

@step(enable_cache=False)
def segmentation_metrics_step(
    checkpoint_path: str,
    metadata_csv: str,
    train_config_path: str = "configs/train_segmentation.yaml",
    threshold: float = 0.5,
    device: str = "cuda",
) -> Annotated[dict, "segmentation_metrics"]:
    """
    Run U-Net on the test set and compute Dice, IoU, pixel AUROC.
    """
    from sklearn.metrics import roc_auc_score
    from models.segmentation import DefectSegmenter
    from data.transforms import get_segmentation_transforms
    from data.dataset import DefectSegmentationDataset
    from torch.utils.data import DataLoader

    device = resolve_torch_device(device, log_warning=logger.warning)
    logger.info(f"Computing segmentation metrics on {device}")

    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    model = DefectSegmenter.load_from_checkpoint(
        checkpoint_path, cfg=train_cfg, map_location=device
    ).to(device).eval()

    transforms = get_segmentation_transforms(512)
    ds = DefectSegmentationDataset(
        csv_path=metadata_csv, split="test",
        transform=transforms["test"], defect_only=True,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    all_probs, all_preds, all_masks = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.sigmoid(logits).squeeze(1)
            preds  = (probs > threshold).long()

            all_probs.append(probs.cpu().numpy().flatten())
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.long().cpu().numpy().flatten())

    probs_flat = np.concatenate(all_probs)
    preds_flat = np.concatenate(all_preds)
    masks_flat = np.concatenate(all_masks)

    tp   = ((preds_flat == 1) & (masks_flat == 1)).sum()
    fp   = ((preds_flat == 1) & (masks_flat == 0)).sum()
    fn   = ((preds_flat == 0) & (masks_flat == 1)).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou  = tp / (tp + fp + fn + 1e-7)

    pixel_auroc = (
        float(roc_auc_score(masks_flat, probs_flat))
        if len(np.unique(masks_flat)) > 1 else 0.0
    )

    results = {
        "dice":        round(float(dice), 4),
        "iou":         round(float(iou), 4),
        "pixel_auroc": round(pixel_auroc, 4),
        "threshold":   threshold,
        "n_pixels":    int(len(masks_flat)),
        "_probs":      probs_flat.tolist(),
        "_labels":     masks_flat.tolist(),
    }

    logger.info(f"Dice={results['dice']:.4f}  IoU={results['iou']:.4f}  Pixel AUROC={results['pixel_auroc']:.4f}")

    log_artifact_metadata(
        metadata={
            "dice":        results["dice"],
            "iou":         results["iou"],
            "pixel_auroc": results["pixel_auroc"],
        }
    )

    return results
