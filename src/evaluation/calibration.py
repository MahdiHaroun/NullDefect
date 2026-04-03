"""
src/evaluation/calibration.py

Calibration curves + all evaluation plots.

What is calibration?
  A model that outputs "85% confidence" should be correct 85% of the time.
  Many neural networks are overconfident — they say 95% when they're right
  only 70% of the time. Calibration curves reveal this gap.

  The reliability diagram plots:
    X axis: predicted probability (binned into 15 buckets)
    Y axis: actual fraction of positives in each bucket
    Perfect calibration = diagonal line
    Above diagonal = underconfident, Below = overconfident

This file also generates:
  - Per-category AUROC bar chart
  - ROC curve
  - Precision-Recall curve
  - Benchmark comparison table (your results vs published PatchCore)

All plots saved as PNG to reports/plots/
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on servers/SageMaker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import yaml


# ── Style ─────────────────────────────────────────────────────────────────────

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

def apply_dark_style(fig, ax_list):
    """Apply consistent dark theme to all plots."""
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["text"], labelsize=9)
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["muted"])


# ── 1. Calibration curve (Reliability Diagram) ───────────────────────────────

def plot_calibration_curve(
    probs: list[float],
    labels: list[int],
    output_dir: str,
    model_name: str = "Classifier",
    n_bins: int = 15,
) -> str:
    """
    Plot reliability diagram showing predicted confidence vs actual accuracy.

    Args:
        probs:      predicted probabilities for positive class
        labels:     ground truth binary labels
        output_dir: directory to save the PNG
        model_name: used in title and filename
        n_bins:     number of calibration bins

    Returns:
        path to saved PNG
    """
    probs  = np.array(probs)
    labels = np.array(labels)

    fraction_pos, mean_pred = calibration_curve(
        labels, probs, n_bins=n_bins, strategy="uniform"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    apply_dark_style(fig, axes)
    fig.suptitle(f"{model_name} — Calibration Analysis",
                 color=STYLE["text"], fontsize=13, fontweight="bold", y=1.01)

    # ── Left: Reliability diagram ──────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color=STYLE["muted"], linewidth=1.5,
            label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color=STYLE["accent"],
            linewidth=2, markersize=6, label=f"{model_name}")

    # Shade the gap between model and perfect calibration
    ax.fill_between(mean_pred, mean_pred, fraction_pos,
                    alpha=0.15, color=STYLE["accent"])

    # Calibration error annotation
    cal_error = np.mean(np.abs(fraction_pos - mean_pred))
    ax.text(0.05, 0.92, f"Mean Cal. Error: {cal_error:.3f}",
            transform=ax.transAxes, color=STYLE["accent"],
            fontsize=9, bbox=dict(boxstyle="round", facecolor=STYLE["bg"], alpha=0.8))

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, labelcolor=STYLE["text"],
              facecolor=STYLE["surface"], edgecolor=STYLE["muted"])

    # ── Right: Confidence histogram ────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(probs[labels == 0], bins=30, alpha=0.7, color=STYLE["blue"],
             label="Normal", density=True)
    ax2.hist(probs[labels == 1], bins=30, alpha=0.7, color=STYLE["red"],
             label="Defective", density=True)
    ax2.axvline(0.5, color=STYLE["muted"], linestyle="--", linewidth=1.2,
                label="Threshold (0.5)")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution")
    ax2.legend(fontsize=9, labelcolor=STYLE["text"],
               facecolor=STYLE["surface"], edgecolor=STYLE["muted"])

    plt.tight_layout()
    out_path = Path(output_dir) / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()

    print(f"  Calibration plot saved: {out_path}")
    return str(out_path)


# ── 2. Per-category AUROC bar chart ──────────────────────────────────────────

def plot_per_category_auroc(
    per_cat_auroc: dict,
    output_dir: str,
    model_name: str = "Classifier",
    published: Optional[dict] = None,
) -> str:
    """
    Horizontal bar chart: one bar per MVTec category showing AUROC.
    Optionally overlays published PatchCore numbers for comparison.

    Args:
        per_cat_auroc: {category: auroc} (from metrics.compute_per_category_auroc)
        published:     {category: auroc} published baseline to compare against
    """
    # Filter out internal keys (start with _)
    cats   = [k for k, v in per_cat_auroc.items()
               if not k.startswith("_") and v is not None]
    aurocs = [per_cat_auroc[c] for c in cats]

    # Sort by AUROC ascending (worst at top → catches eye)
    sorted_pairs = sorted(zip(aurocs, cats), key=lambda x: x[0])
    aurocs_sorted = [p[0] for p in sorted_pairs]
    cats_sorted   = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, max(6, len(cats) * 0.45)))
    apply_dark_style(fig, ax)

    y = np.arange(len(cats_sorted))

    # Your results
    bars = ax.barh(y, aurocs_sorted, height=0.5,
                   color=STYLE["accent"], alpha=0.85, label="Your model")

    # Published baseline (if provided)
    if published:
        pub_vals = [published.get(c, 0) for c in cats_sorted]
        ax.barh(y, pub_vals, height=0.5, left=0,
                color=STYLE["muted"], alpha=0.35, label="Published (PatchCore)")

    # Value labels on bars
    for bar, val in zip(bars, aurocs_sorted):
        color = STYLE["green"] if val >= 0.95 else (STYLE["accent"] if val >= 0.90 else STYLE["red"])
        ax.text(min(val + 0.002, 0.998), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color=color, fontsize=8, fontweight="bold")

    # Mean line
    mean_auroc = per_cat_auroc.get("_mean", np.mean(aurocs_sorted))
    ax.axvline(mean_auroc, color=STYLE["green"], linestyle="--",
               linewidth=1.5, label=f"Mean: {mean_auroc:.3f}")

    ax.set_yticks(y)
    ax.set_yticklabels(cats_sorted, fontsize=9)
    ax.set_xlabel("AUROC")
    ax.set_title(f"{model_name} — Per-Category AUROC (MVTec AD)",
                 fontweight="bold", pad=12)
    ax.set_xlim(0.5, 1.02)
    ax.legend(fontsize=9, labelcolor=STYLE["text"],
              facecolor=STYLE["surface"], edgecolor=STYLE["muted"],
              loc="lower right")

    plt.tight_layout()
    out_path = Path(output_dir) / f"auroc_per_category_{model_name.lower().replace(' ', '_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()

    print(f"  Per-category AUROC chart saved: {out_path}")
    return str(out_path)


# ── 3. ROC Curve ─────────────────────────────────────────────────────────────

def plot_roc_curve(
    probs: list[float],
    labels: list[int],
    output_dir: str,
    model_name: str = "Classifier",
) -> str:
    probs  = np.array(probs)
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    apply_dark_style(fig, ax)

    ax.plot(fpr, tpr, color=STYLE["accent"], linewidth=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=STYLE["muted"],
            linewidth=1.2, label="Random (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color=STYLE["accent"])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve", fontweight="bold")
    ax.legend(fontsize=10, labelcolor=STYLE["text"],
              facecolor=STYLE["surface"], edgecolor=STYLE["muted"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    out_path = Path(output_dir) / f"roc_{model_name.lower().replace(' ', '_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()

    print(f"  ROC curve saved: {out_path}")
    return str(out_path)


# ── 4. Benchmark comparison table ─────────────────────────────────────────────

def plot_benchmark_comparison(
    anomaly_results: dict,
    output_dir: str,
) -> str:
    """
    Visual table comparing your PatchCore results vs published numbers.
    Each row = one MVTec category.
    Color-coded: green = within 1% of published, amber = within 3%, red = worse.
    """
    per_cat = anomaly_results["per_category"]
    categories = sorted(per_cat.keys())

    your_vals = [per_cat[c]["your_image_auroc"] or 0.0  for c in categories]
    pub_vals  = [per_cat[c]["published_image_auroc"]     for c in categories]
    deltas    = [per_cat[c]["delta"] or 0.0              for c in categories]

    fig, ax = plt.subplots(figsize=(11, max(6, len(categories) * 0.52)))
    apply_dark_style(fig, ax)
    ax.axis("off")

    # Table data
    col_labels = ["Category", "Your AUROC", "Published", "Delta"]
    table_data = []
    cell_colors = []

    for cat, your, pub, delta in zip(categories, your_vals, pub_vals, deltas):
        delta_str = f"{delta:+.4f}" if delta != 0 else "N/A"
        table_data.append([cat, f"{your:.4f}", f"{pub:.4f}", delta_str])

        if delta >= -0.005:
            d_color = "#14532d"    # dark green — within 0.5%
        elif delta >= -0.02:
            d_color = "#78350f"    # dark amber — within 2%
        else:
            d_color = "#7f1d1d"    # dark red — worse by >2%

        cell_colors.append([STYLE["surface"], STYLE["surface"], STYLE["surface"], d_color])

    # Add mean row
    mean_your  = anomaly_results["mean_image_auroc"]
    mean_pub   = anomaly_results["published_mean_image_auroc"]
    mean_delta = anomaly_results["mean_delta"]
    table_data.append(["MEAN", f"{mean_your:.4f}", f"{mean_pub:.4f}", f"{mean_delta:+.4f}"])
    mean_d_color = "#14532d" if mean_delta >= -0.005 else "#78350f"
    cell_colors.append(["#1f2937", "#1f2937", "#1f2937", mean_d_color])

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Style header
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor(STYLE["accent"])
        tbl[0, col].set_text_props(color="#000000", fontweight="bold")

    # Style all text
    for (row, col), cell in tbl.get_celld().items():
        if row > 0:
            cell.set_text_props(color=STYLE["text"])

    fig.suptitle(
        "PatchCore Benchmark: Your Results vs Published (Roth et al., 2022)",
        color=STYLE["text"], fontsize=12, fontweight="bold", y=0.98,
    )

    plt.tight_layout()
    out_path = Path(output_dir) / "benchmark_comparison_patchcore.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()

    print(f"  Benchmark table saved: {out_path}")
    return str(out_path)
