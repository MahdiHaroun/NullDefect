"""
src/evaluation/report.py

Evidently AI HTML report generator.

What Evidently does:
  Takes your model's predictions and ground truth labels,
  runs a suite of statistical tests and visualizations,
  and produces a self-contained HTML report you can open in any browser.

We generate three reports:
  1. Classification report  — for the EfficientNet-B4 classifier
  2. Data quality report    — for the full dataset (missing values, distributions)
  3. Model comparison       — FP32 vs quantized model (used in Phase 4)

Evidently install: pip install evidently
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/evaluate.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── 1. Classification report ──────────────────────────────────────────────────

def generate_classification_report(
    probs: list[float],
    labels: list[int],
    output_dir: str,
    model_name: str = "EfficientNet-B4 Classifier",
    threshold: float = 0.5,
) -> str:
    """
    Generates an Evidently classification performance report.

    Includes:
      - Accuracy, Precision, Recall, F1, ROC AUC
      - Confusion matrix heatmap
      - Class separation histogram
      - Precision-Recall curve
      - Probability distribution per class

    Args:
        probs:      model predicted probabilities for positive class [N]
        labels:     ground truth binary labels [N]
        output_dir: where to save the HTML file
        threshold:  decision threshold (default 0.5)

    Returns:
        path to saved HTML report
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset
        from evidently.metrics import (
            ClassificationQualityMetric,
            ClassificationClassBalance,
            ClassificationConfusionMatrix,
            ClassificationRocCurve,
            ClassificationPRCurve,
            ClassificationProbDistribution,
            ClassificationQualityByClass,
        )
    except ImportError:
        raise ImportError("evidently not installed. Run: pip install evidently")

    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= threshold).astype(int)

    # Evidently expects a DataFrame with specific column names
    df = pd.DataFrame({
        "target":     labels,
        "prediction": preds,
        "Normal":     1 - probs,      # prob of class 0
        "Defective":  probs,          # prob of class 1
    })

    # Evidently compares a "current" dataset to a "reference" dataset.
    # Here we use the full test set as current and no reference (single-model report).
    report = Report(metrics=[
        ClassificationPreset(probas_threshold=threshold),
    ])

    report.run(
        current_data=df,
        reference_data=None,
        column_mapping=_build_column_mapping(),
    )

    out_path = Path(output_dir) / f"classification_report_{model_name.lower().replace(' ', '_').replace('-', '_')}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))

    print(f"  Evidently classification report saved: {out_path}")
    return str(out_path)


def _build_column_mapping():
    """Evidently ColumnMapping tells the library which columns are which."""
    try:
        from evidently import ColumnMapping
    except ImportError:
        return None

    return ColumnMapping(
        target="target",
        prediction="prediction",
        prediction_probas=["Normal", "Defective"],
    )


# ── 2. Data quality report ────────────────────────────────────────────────────

def generate_data_quality_report(
    metadata_csv_path: str,
    output_dir: str,
) -> str:
    """
    Generates an Evidently data quality report for the full dataset.

    Checks:
      - Missing values per column
      - Value distributions (label, category, defect_type, split)
      - Class balance per split
      - Dataset statistics summary
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataQualityPreset
    except ImportError:
        raise ImportError("evidently not installed. Run: pip install evidently")

    df = pd.read_csv(metadata_csv_path)

    # Sample for speed (Evidently can be slow on very large DataFrames)
    sample_size = min(10_000, len(df))
    df_sample = df.sample(sample_size, random_state=42)

    report = Report(metrics=[DataQualityPreset()])
    report.run(current_data=df_sample, reference_data=None)

    out_path = Path(output_dir) / "data_quality_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))

    print(f"  Evidently data quality report saved: {out_path}")
    return str(out_path)


# ── 3. Model comparison report (used in Phase 4 — Quantization) ──────────────

def generate_model_comparison_report(
    fp32_probs: list[float],
    quantized_probs: list[float],
    labels: list[int],
    output_dir: str,
    quantization_strategy: str = "PTQ INT8",
) -> str:
    """
    Compares FP32 and quantized model predictions using Evidently.

    Used in Phase 4 to show that quantization preserves model behavior.
    The "drift" between FP32 and quantized predictions should be near zero.

    Treats:
      reference_data = FP32 predictions
      current_data   = quantized model predictions

    Metrics shown:
      - Distribution shift between FP32 and quantized probabilities
      - Agreement rate (% of samples where both models make same prediction)
      - Accuracy comparison
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset
        from evidently.metrics import ClassificationQualityMetric
    except ImportError:
        raise ImportError("evidently not installed. Run: pip install evidently")

    fp32_preds = (np.array(fp32_probs) >= 0.5).astype(int)
    quant_preds = (np.array(quantized_probs) >= 0.5).astype(int)

    df_fp32 = pd.DataFrame({
        "target":     labels,
        "prediction": fp32_preds,
        "Normal":     1 - np.array(fp32_probs),
        "Defective":  np.array(fp32_probs),
    })

    df_quant = pd.DataFrame({
        "target":     labels,
        "prediction": quant_preds,
        "Normal":     1 - np.array(quantized_probs),
        "Defective":  np.array(quantized_probs),
    })

    report = Report(metrics=[
        ClassificationPreset(),
        ClassificationQualityMetric(),
    ])

    report.run(
        reference_data=df_fp32,
        current_data=df_quant,
        column_mapping=_build_column_mapping(),
    )

    strategy_slug = quantization_strategy.lower().replace(" ", "_")
    out_path = Path(output_dir) / f"model_comparison_fp32_vs_{strategy_slug}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))

    # Agreement rate (useful stat for the quantization report)
    agreement = float(np.mean(fp32_preds == quant_preds))
    print(f"  FP32 vs {quantization_strategy} agreement rate: {agreement:.4f}")
    print(f"  Evidently comparison report saved: {out_path}")
    return str(out_path)


# ── 4. Segmentation report ────────────────────────────────────────────────────

def generate_segmentation_report(
    pixel_probs: list[float],
    pixel_labels: list[int],
    output_dir: str,
    threshold: float = 0.5,
) -> str:
    """
    Evidently report for segmentation quality at the pixel level.
    Treats pixel-level predictions as a binary classification problem.
    Samples 100K pixels for speed.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset
    except ImportError:
        raise ImportError("evidently not installed.")

    probs  = np.array(pixel_probs)
    labels = np.array(pixel_labels)

    # Sample pixels for speed
    if len(probs) > 100_000:
        idx    = np.random.choice(len(probs), 100_000, replace=False)
        probs  = probs[idx]
        labels = labels[idx]

    preds = (probs >= threshold).astype(int)
    df = pd.DataFrame({
        "target":     labels,
        "prediction": preds,
        "Normal":     1 - probs,
        "Defective":  probs,
    })

    report = Report(metrics=[ClassificationPreset(probas_threshold=threshold)])
    report.run(current_data=df, reference_data=None,
               column_mapping=_build_column_mapping())

    out_path = Path(output_dir) / "segmentation_pixel_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))

    print(f"  Evidently segmentation report saved: {out_path}")
    return str(out_path)
