"""
steps/evaluation/evidently_step.py

ZenML step — generate Evidently AI HTML reports.

Three reports:
  1. Classification performance report (EfficientNet-B4)
  2. Data quality report (full dataset)
  3. Segmentation pixel-level report (U-Net)

Evidently install: uv add evidently
"""

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

logger = get_logger(__name__)


def _build_column_mapping():
    try:
        from evidently import ColumnMapping
        return ColumnMapping(
            target="target",
            prediction="prediction",
            prediction_probas=["Normal", "Defective"],
        )
    except Exception:
        return None


@step(enable_cache=False)
def evidently_reports_step(
    classifier_metrics: dict,
    segmentation_metrics: dict,
    metadata_csv: str,
    output_dir: str = "reports/evidently",
) -> Annotated[dict, "evidently_report_paths"]:
    """
    Generate all Evidently HTML reports.
    Returns dict of report name → file path.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset, DataQualityPreset
    except ImportError:
        logger.warning("evidently not installed. Run: uv add evidently")
        logger.warning("Skipping Evidently reports step.")
        return {"skipped": True, "reason": "evidently not installed"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    col_map = _build_column_mapping()

    # ── 1. Classification performance report ───────────────────────────────
    logger.info("Generating classification performance report...")
    probs  = np.array(classifier_metrics.get("_probs", []))
    labels = np.array(classifier_metrics.get("_labels", []))

    if len(probs) > 0:
        threshold = classifier_metrics.get("threshold", 0.5)
        preds     = (probs >= threshold).astype(int)

        clf_df = pd.DataFrame({
            "target":     labels,
            "prediction": preds,
            "Normal":     1 - probs,
            "Defective":  probs,
        })

        try:
            report = Report(metrics=[ClassificationPreset(probas_threshold=threshold)])
            report.run(current_data=clf_df, reference_data=None,
                       column_mapping=col_map)

            clf_path = out / "classification_report_efficientnet_b4.html"
            report.save_html(str(clf_path))
            saved_paths["classification"] = str(clf_path)
            logger.info(f"Classification report saved: {clf_path}")
        except Exception as e:
            logger.warning(f"Classification report failed: {e}")

    # ── 2. Data quality report ─────────────────────────────────────────────
    logger.info("Generating data quality report...")
    try:
        df = pd.read_csv(metadata_csv)
        # Sample for speed
        sample = df.sample(min(10_000, len(df)), random_state=42)

        report = Report(metrics=[DataQualityPreset()])
        report.run(current_data=sample, reference_data=None)

        dq_path = out / "data_quality_report.html"
        report.save_html(str(dq_path))
        saved_paths["data_quality"] = str(dq_path)
        logger.info(f"Data quality report saved: {dq_path}")
    except Exception as e:
        logger.warning(f"Data quality report failed: {e}")

    # ── 3. Segmentation pixel-level report ─────────────────────────────────
    logger.info("Generating segmentation pixel report...")
    seg_probs  = np.array(segmentation_metrics.get("_probs", []))
    seg_labels = np.array(segmentation_metrics.get("_labels", []))

    if len(seg_probs) > 0:
        # Sample pixels for speed
        if len(seg_probs) > 100_000:
            idx        = np.random.choice(len(seg_probs), 100_000, replace=False)
            seg_probs  = seg_probs[idx]
            seg_labels = seg_labels[idx]

        threshold  = segmentation_metrics.get("threshold", 0.5)
        seg_preds  = (seg_probs >= threshold).astype(int)

        seg_df = pd.DataFrame({
            "target":     seg_labels,
            "prediction": seg_preds,
            "Normal":     1 - seg_probs,
            "Defective":  seg_probs,
        })

        try:
            report = Report(metrics=[ClassificationPreset(probas_threshold=threshold)])
            report.run(current_data=seg_df, reference_data=None,
                       column_mapping=col_map)

            seg_path = out / "segmentation_pixel_report.html"
            report.save_html(str(seg_path))
            saved_paths["segmentation"] = str(seg_path)
            logger.info(f"Segmentation report saved: {seg_path}")
        except Exception as e:
            logger.warning(f"Segmentation report failed: {e}")

    logger.info(f"Evidently reports saved to {out.resolve()}")
    return saved_paths
