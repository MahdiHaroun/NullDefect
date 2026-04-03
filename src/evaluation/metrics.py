"""
src/evaluation/metrics.py

Evaluation metrics for all three models.

  ClassifierMetrics   → EfficientNet-B4 (AUROC, F1, precision, recall, confusion matrix)
  AnomalyMetrics      → PatchCore (image AUROC, pixel AUROC, per-category breakdown)
  SegmentationMetrics → U-Net (Dice, IoU, pixel AUROC)

All metric functions return plain dicts — easy to log to MLflow,
serialize to JSON, or feed into the report generator.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)
import yaml


def load_config(path: str = "configs/evaluate.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def load_checkpoint_or_s3(local_path: str, cfg: dict):
    """
    Load checkpoint from local disk. If not found, download from S3.
    Returns the resolved local path.
    """
    p = Path(local_path)
    if p.exists():
        return str(p)

    # Try S3 fallback
    import boto3, os
    bucket = os.environ.get("SAGEMAKER_S3_BUCKET", cfg["checkpoints"]["s3_bucket"])
    s3_prefix = cfg["checkpoints"]["s3_prefix"]
    model_key = p.name
    s3_key = f"{s3_prefix}/{p.parent.name}/{model_key}"

    print(f"  [S3] Checkpoint not found locally. Downloading from s3://{bucket}/{s3_key}")
    p.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    s3.download_file(bucket, s3_key, str(p))
    print(f"  [S3] Downloaded to {p}")
    return str(p)


# ── 1. Classifier Metrics ─────────────────────────────────────────────────────

class ClassifierMetrics:
    """
    Runs the EfficientNet-B4 classifier on the test set and computes:
      - Binary AUROC (overall)
      - F1, precision, recall at the configured threshold
      - Confusion matrix
      - Per-category AUROC (run separately per MVTec category)
      - Average Precision (area under precision-recall curve)
    """

    def __init__(self, model, dataloader: DataLoader, device: str, threshold: float = 0.5):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def collect_predictions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass on the full test set.
        Returns (all_probs, all_preds, all_labels) as numpy arrays.
        """
        all_probs, all_preds, all_labels = [], [], []

        for images, labels in self.dataloader:
            images = images.to(self.device)
            logits = self.model(images)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs >= self.threshold).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

        return (
            np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels),
        )

    def compute(self) -> dict:
        """Compute and return all classification metrics."""
        print("  Computing classifier metrics...")
        probs, preds, labels = self.collect_predictions()

        # Guard: skip AUROC if only one class present in test set
        unique_labels = np.unique(labels)
        auroc = (
            roc_auc_score(labels, probs)
            if len(unique_labels) > 1 else 0.0
        )

        results = {
            "auroc":            round(float(auroc), 4),
            "f1":               round(float(f1_score(labels, preds, zero_division=0)), 4),
            "precision":        round(float(precision_score(labels, preds, zero_division=0)), 4),
            "recall":           round(float(recall_score(labels, preds, zero_division=0)), 4),
            "avg_precision":    round(float(average_precision_score(labels, probs)), 4),
            "threshold":        self.threshold,
            "n_samples":        len(labels),
            "n_normal":         int((labels == 0).sum()),
            "n_defective":      int((labels == 1).sum()),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            # Store raw arrays for calibration and plotting
            "_probs":           probs.tolist(),
            "_labels":          labels.tolist(),
        }

        print(f"    AUROC={results['auroc']:.4f}  F1={results['f1']:.4f}  "
              f"Precision={results['precision']:.4f}  Recall={results['recall']:.4f}")
        return results


# ── Per-category AUROC ────────────────────────────────────────────────────────

def compute_per_category_auroc(
    model,
    metadata_csv_path: str,
    transforms: dict,
    cfg: dict,
    device: str,
) -> dict:
    """
    Compute classifier AUROC separately for each of the 15 MVTec categories.

    This tells you which categories are hard (low AUROC) vs easy (high AUROC)
    — important for understanding model weaknesses.

    Returns dict: {category: auroc_score}
    """
    import pandas as pd
    import sys
    sys.path.insert(0, "src")
    from data.dataset import DefectClassificationDataset
    from torch.utils.data import DataLoader

    data_cfg = yaml.safe_load(open("configs/data.yaml"))
    categories = data_cfg["dataset"]["categories"]

    model = model.to(device).eval()
    per_cat_auroc = {}

    for category in categories:
        ds = DefectClassificationDataset(
            csv_path=metadata_csv_path,
            split="test",
            transform=transforms["test"],
            category=category,
        )

        if len(ds) == 0:
            print(f"    [{category}] No test samples — skipping")
            continue

        loader = DataLoader(ds, batch_size=cfg["evaluation"]["batch_size"],
                            shuffle=False, num_workers=2)

        probs_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs_list.append(probs)
                labels_list.append(labels.numpy())

        probs_cat  = np.concatenate(probs_list)
        labels_cat = np.concatenate(labels_list)

        unique = np.unique(labels_cat)
        if len(unique) < 2:
            # Category only has normal samples in test — AUROC undefined
            per_cat_auroc[category] = None
            continue

        auroc = roc_auc_score(labels_cat, probs_cat)
        per_cat_auroc[category] = round(float(auroc), 4)
        print(f"    [{category:<16}] AUROC={auroc:.4f}  "
              f"(n={len(labels_cat)}, defects={int(labels_cat.sum())})")

    valid = [v for v in per_cat_auroc.values() if v is not None]
    mean_auroc = round(float(np.mean(valid)), 4) if valid else 0.0
    per_cat_auroc["_mean"] = mean_auroc
    print(f"\n    Mean AUROC across categories: {mean_auroc:.4f}")

    return per_cat_auroc


# ── 2. Anomaly Metrics ────────────────────────────────────────────────────────

class AnomalyMetrics:
    """
    Collects PatchCore results from the anomalib results JSON files
    and computes the benchmark comparison table vs published numbers.
    """

    # Published PatchCore numbers (Roth et al., 2022)
    PUBLISHED = {
        "bottle":      0.9996, "cable":    0.9938, "capsule":  0.9820,
        "carpet":      0.9870, "grid":     0.9827, "hazelnut": 1.0000,
        "leather":     1.0000, "metal_nut":1.0000, "pill":     0.9661,
        "screw":       0.9750, "tile":     0.9874, "toothbrush":1.0000,
        "transistor":  1.0000, "wood":     0.9891, "zipper":   0.9985,
    }

    def __init__(self, results_json_path: str):
        with open(results_json_path) as f:
            self.raw = json.load(f)

    def compute(self) -> dict:
        """
        Build comparison table: your results vs published PatchCore.
        Returns dict with per_category comparison and mean metrics.
        """
        print("  Computing anomaly detection metrics...")
        per_cat = self.raw.get("per_category", {})

        comparison = {}
        for cat, published_auroc in self.PUBLISHED.items():
            your_auroc = per_cat.get(cat, {}).get("image_auroc")
            delta = round(your_auroc - published_auroc, 4) if your_auroc else None
            comparison[cat] = {
                "your_image_auroc":      your_auroc,
                "published_image_auroc": published_auroc,
                "delta":                 delta,
                "your_pixel_auroc":      per_cat.get(cat, {}).get("pixel_auroc"),
            }

        your_aurocs = [
            v["your_image_auroc"] for v in comparison.values()
            if v["your_image_auroc"] is not None
        ]
        mean_your  = round(float(np.mean(your_aurocs)), 4) if your_aurocs else 0.0
        mean_pub   = round(float(np.mean(list(self.PUBLISHED.values()))), 4)

        results = {
            "mean_image_auroc":           mean_your,
            "mean_pixel_auroc":           self.raw.get("mean_pixel_auroc", 0.0),
            "published_mean_image_auroc": mean_pub,
            "mean_delta":                 round(mean_your - mean_pub, 4),
            "per_category":               comparison,
        }

        print(f"    Your mean AUROC     : {mean_your:.4f}")
        print(f"    Published mean AUROC: {mean_pub:.4f}")
        print(f"    Delta               : {results['mean_delta']:+.4f}")
        return results


# ── 3. Segmentation Metrics ───────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Runs the U-Net segmenter on the test set and computes:
      - Dice coefficient
      - IoU (Intersection over Union)
      - Pixel-level AUROC
    """

    def __init__(self, model, dataloader: DataLoader, device: str, threshold: float = 0.5):
        self.model     = model.to(device).eval()
        self.dataloader = dataloader
        self.device    = device
        self.threshold = threshold

    @torch.no_grad()
    def compute(self) -> dict:
        print("  Computing segmentation metrics...")
        all_probs, all_preds, all_masks = [], [], []

        for images, masks in self.dataloader:
            images = images.to(self.device)
            logits = self.model(images)                        # [B,1,H,W]
            probs  = torch.sigmoid(logits).squeeze(1)         # [B,H,W]
            preds  = (probs > self.threshold).long()

            all_probs.append(probs.cpu().numpy().flatten())
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.long().cpu().numpy().flatten())

        probs_flat = np.concatenate(all_probs)
        preds_flat = np.concatenate(all_preds)
        masks_flat = np.concatenate(all_masks)

        # Dice coefficient: 2*TP / (2*TP + FP + FN)
        tp = ((preds_flat == 1) & (masks_flat == 1)).sum()
        fp = ((preds_flat == 1) & (masks_flat == 0)).sum()
        fn = ((preds_flat == 0) & (masks_flat == 1)).sum()
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)

        # IoU: TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-7)

        # Pixel AUROC
        pixel_auroc = (
            roc_auc_score(masks_flat, probs_flat)
            if len(np.unique(masks_flat)) > 1 else 0.0
        )

        results = {
            "dice":        round(float(dice), 4),
            "iou":         round(float(iou), 4),
            "pixel_auroc": round(float(pixel_auroc), 4),
            "threshold":   self.threshold,
            "n_pixels":    len(masks_flat),
            "_probs":      probs_flat.tolist(),
            "_labels":     masks_flat.tolist(),
        }

        print(f"    Dice={results['dice']:.4f}  "
              f"IoU={results['iou']:.4f}  "
              f"Pixel AUROC={results['pixel_auroc']:.4f}")
        return results
