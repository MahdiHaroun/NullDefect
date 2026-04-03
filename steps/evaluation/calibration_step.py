"""
steps/evaluation/calibration_step.py

ZenML step — generate all evaluation plots:
  - Calibration curve (reliability diagram + confidence distribution)
  - Per-category AUROC bar chart (classifier + PatchCore vs published)
  - ROC curve
  - PatchCore benchmark comparison table
"""

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zenml import step
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

logger = get_logger(__name__)

# Dark theme
STYLE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "accent":  "#f59e0b",
    "blue":    "#3b82f6",
    "green":   "#22c55e",
    "red":     "#ef4444",
    "muted":   "#6b7280",
    "text":    "#e2e8f0",
}


def _style_ax(fig, axes):
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in (axes if isinstance(axes, (list, np.ndarray)) else [axes]):
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["text"], labelsize=9)
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["muted"])


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return str(path)


@step(enable_cache=False)
def calibration_plots_step(
    classifier_metrics: dict,
    anomaly_metrics: dict,
    output_dir: str = "reports/plots",
) -> Annotated[dict, "plot_paths"]:
    """
    Generate all evaluation plots from precomputed metrics.
    Returns dict of plot name → file path.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve, auc

    out = Path(output_dir)
    saved_paths = {}

    probs  = np.array(classifier_metrics.get("_probs", []))
    labels = np.array(classifier_metrics.get("_labels", []))

    if len(probs) == 0:
        logger.warning("No classifier predictions available for plots")
        return saved_paths

    # ── 1. Calibration curve ───────────────────────────────────────────────
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=15, strategy="uniform")
    cal_error = float(np.mean(np.abs(fraction_pos - mean_pred)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _style_ax(fig, axes)
    fig.suptitle("EfficientNet-B4 — Calibration Analysis",
                 color=STYLE["text"], fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color=STYLE["muted"], lw=1.5, label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color=STYLE["accent"], lw=2, ms=6, label="Model")
    ax.fill_between(mean_pred, mean_pred, fraction_pos, alpha=0.12, color=STYLE["accent"])
    ax.text(0.05, 0.92, f"Mean Cal. Error: {cal_error:.3f}",
            transform=ax.transAxes, color=STYLE["accent"], fontsize=9,
            bbox=dict(boxstyle="round", facecolor=STYLE["bg"], alpha=0.8))
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, labelcolor=STYLE["text"],
              facecolor=STYLE["surface"], edgecolor=STYLE["muted"])

    ax2 = axes[1]
    ax2.hist(probs[labels == 0], bins=30, alpha=0.7, color=STYLE["blue"],
             label="Normal", density=True)
    ax2.hist(probs[labels == 1], bins=30, alpha=0.7, color=STYLE["red"],
             label="Defective", density=True)
    ax2.axvline(0.5, color=STYLE["muted"], linestyle="--", lw=1.2, label="Threshold")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution")
    ax2.legend(fontsize=9, labelcolor=STYLE["text"],
               facecolor=STYLE["surface"], edgecolor=STYLE["muted"])

    plt.tight_layout()
    saved_paths["calibration"] = _save(fig, out / "calibration_efficientnet_b4.png")

    # ── 2. ROC Curve ───────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    _style_ax(fig, ax)
    ax.plot(fpr, tpr, color=STYLE["accent"], lw=2, label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=STYLE["muted"], lw=1.2, label="Random (AUC=0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color=STYLE["accent"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("EfficientNet-B4 — ROC Curve", fontweight="bold")
    ax.legend(fontsize=10, labelcolor=STYLE["text"],
              facecolor=STYLE["surface"], edgecolor=STYLE["muted"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    saved_paths["roc"] = _save(fig, out / "roc_classifier.png")

    # ── 3. Per-category AUROC bar chart ────────────────────────────────────
    per_cat = classifier_metrics.get("per_category_auroc", {})
    cats    = [k for k, v in per_cat.items() if not k.startswith("_") and v is not None]
    aurocs  = [per_cat[c] for c in cats]
    sorted_pairs = sorted(zip(aurocs, cats))
    aurocs_s     = [p[0] for p in sorted_pairs]
    cats_s       = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, max(6, len(cats) * 0.45)))
    _style_ax(fig, ax)
    y    = np.arange(len(cats_s))
    bars = ax.barh(y, aurocs_s, height=0.5, color=STYLE["accent"], alpha=0.85,
                   label="Your model")
    for bar, val in zip(bars, aurocs_s):
        color = STYLE["green"] if val >= 0.95 else (STYLE["accent"] if val >= 0.90 else STYLE["red"])
        ax.text(min(val + 0.002, 0.998), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", color=color, fontsize=8, fontweight="bold")

    mean_auroc = per_cat.get("_mean", np.mean(aurocs_s))
    ax.axvline(mean_auroc, color=STYLE["green"], linestyle="--", lw=1.5,
               label=f"Mean: {mean_auroc:.3f}")
    ax.set_yticks(y); ax.set_yticklabels(cats_s, fontsize=9)
    ax.set_xlabel("AUROC")
    ax.set_title("Classifier — Per-Category AUROC (MVTec AD)", fontweight="bold", pad=12)
    ax.set_xlim(0.5, 1.02)
    ax.legend(fontsize=9, labelcolor=STYLE["text"],
              facecolor=STYLE["surface"], edgecolor=STYLE["muted"], loc="lower right")
    plt.tight_layout()
    saved_paths["per_cat_classifier"] = _save(fig, out / "auroc_per_category_classifier.png")

    # ── 4. PatchCore benchmark comparison ──────────────────────────────────
    if anomaly_metrics and "per_category" in anomaly_metrics:
        PUBLISHED = {
            "bottle": 0.9996, "cable": 0.9938, "capsule": 0.9820,
            "carpet": 0.9870, "grid": 0.9827, "hazelnut": 1.0000,
            "leather": 1.0000, "metal_nut": 1.0000, "pill": 0.9661,
            "screw": 0.9750, "tile": 0.9874, "toothbrush": 1.0000,
            "transistor": 1.0000, "wood": 0.9891, "zipper": 0.9985,
        }
        per_anom = anomaly_metrics["per_category"]
        a_cats   = sorted(per_anom.keys())
        a_yours  = [per_anom[c].get("your_image_auroc") or 0 for c in a_cats]
        a_pub    = [PUBLISHED.get(c, 0) for c in a_cats]

        fig, ax = plt.subplots(figsize=(10, max(6, len(a_cats) * 0.45)))
        _style_ax(fig, ax)
        x = np.arange(len(a_cats))
        ax.barh(x - 0.2, a_yours, height=0.35, color=STYLE["accent"],
                label="Your PatchCore")
        ax.barh(x + 0.2, a_pub,   height=0.35, color=STYLE["muted"],
                alpha=0.5, label="Published (Roth et al., 2022)")
        ax.set_yticks(x); ax.set_yticklabels(a_cats, fontsize=9)
        ax.set_xlabel("Image AUROC")
        ax.set_title("PatchCore — Your Results vs Published", fontweight="bold")
        mean_y = anomaly_metrics.get("mean_image_auroc", 0)
        ax.axvline(mean_y, color=STYLE["green"], linestyle="--", lw=1.5,
                   label=f"Your mean: {mean_y:.4f}")
        ax.legend(fontsize=9, labelcolor=STYLE["text"],
                  facecolor=STYLE["surface"], edgecolor=STYLE["muted"])
        ax.set_xlim(0.5, 1.02)
        plt.tight_layout()
        saved_paths["patchcore_benchmark"] = _save(fig, out / "auroc_per_category_patchcore.png")

    logger.info(f"All plots saved to {out.resolve()}")
    return saved_paths
