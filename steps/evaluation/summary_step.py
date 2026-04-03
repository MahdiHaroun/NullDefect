"""
steps/evaluation/summary_step.py

ZenML step — generate the final evaluation summary.

Combines all metrics into:
  1. evaluation_results.json — all metrics in one file
  2. summary printed to console
  3. All metrics logged to ZenML dashboard + MLflow
"""

import json
import sys
from pathlib import Path
from typing import Annotated

from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

logger = get_logger(__name__)


@step(enable_cache=False)
def summary_step(
    classifier_metrics: dict,
    anomaly_metrics: dict,
    segmentation_metrics: dict,
    plot_paths: dict,
    report_paths: dict,
    output_dir: str = "reports",
) -> Annotated[dict, "evaluation_summary"]:
    """
    Combine all metrics, save JSON summary, log to ZenML + MLflow.
    """
    import mlflow

    # ── Build summary ──────────────────────────────────────────────────────
    summary = {
        "classifier": {
            k: v for k, v in classifier_metrics.items()
            if not k.startswith("_")   # exclude raw probs/labels arrays
        },
        "anomaly":      anomaly_metrics,
        "segmentation": {
            k: v for k, v in segmentation_metrics.items()
            if not k.startswith("_")
        },
        "plots":   plot_paths,
        "reports": report_paths,
    }

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_path = Path(output_dir) / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation results saved: {out_path}")

    # ── Log to MLflow ──────────────────────────────────────────────────────
    try:
        mlflow.set_experiment("nulldefect_evaluation")
        with mlflow.start_run(run_name="phase3_evaluation"):
            # Classifier
            mlflow.log_metrics({
                "clf_auroc":           classifier_metrics.get("auroc", 0),
                "clf_f1":              classifier_metrics.get("f1", 0),
                "clf_precision":       classifier_metrics.get("precision", 0),
                "clf_recall":          classifier_metrics.get("recall", 0),
                "clf_mean_cat_auroc":  classifier_metrics.get(
                    "per_category_auroc", {}).get("_mean", 0),
            })

            # Anomaly
            mlflow.log_metrics({
                "anomaly_mean_image_auroc":  anomaly_metrics.get("mean_image_auroc", 0),
                "anomaly_mean_pixel_auroc":  anomaly_metrics.get("mean_pixel_auroc", 0),
                "anomaly_delta_vs_published": anomaly_metrics.get("mean_delta", 0),
            })

            # Segmentation
            mlflow.log_metrics({
                "seg_dice":        segmentation_metrics.get("dice", 0),
                "seg_iou":         segmentation_metrics.get("iou", 0),
                "seg_pixel_auroc": segmentation_metrics.get("pixel_auroc", 0),
            })

            # Log artifact dirs
            for artifact_dir in [
                Path(output_dir) / "plots",
                Path(output_dir) / "evidently",
                Path(output_dir) / "gradcam",
            ]:
                if artifact_dir.exists():
                    mlflow.log_artifacts(str(artifact_dir))

        logger.info("Metrics logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    # ── Log to ZenML dashboard ─────────────────────────────────────────────
    log_artifact_metadata(
        metadata={
            # Classifier
            "clf_auroc":     classifier_metrics.get("auroc", 0),
            "clf_f1":        classifier_metrics.get("f1", 0),
            # Anomaly
            "anomaly_auroc": anomaly_metrics.get("mean_image_auroc", 0),
            "anomaly_delta": anomaly_metrics.get("mean_delta", 0),
            # Segmentation
            "seg_dice":      segmentation_metrics.get("dice", 0),
            "seg_iou":       segmentation_metrics.get("iou", 0),
        }
    )

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  NULLDEFECT — PHASE 3 EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  {'─'*51}")
    print(f"  {'MODEL':<30} {'METRIC':<15} {'SCORE'}")
    print(f"  {'─'*51}")
    print(f"  {'EfficientNet-B4 Classifier':<30} {'AUROC':<15} {classifier_metrics.get('auroc', 'N/A')}")
    print(f"  {'EfficientNet-B4 Classifier':<30} {'F1':<15} {classifier_metrics.get('f1', 'N/A')}")
    print(f"  {'─'*51}")
    print(f"  {'PatchCore Anomaly':<30} {'Image AUROC':<15} {anomaly_metrics.get('mean_image_auroc', 'N/A')}")
    print(f"  {'PatchCore Anomaly':<30} {'Δ Published':<15} {anomaly_metrics.get('mean_delta', 'N/A')}")
    print(f"  {'─'*51}")
    print(f"  {'U-Net Segmenter':<30} {'Dice':<15} {segmentation_metrics.get('dice', 'N/A')}")
    print(f"  {'U-Net Segmenter':<30} {'IoU':<15} {segmentation_metrics.get('iou', 'N/A')}")
    print(f"  {'─'*51}")
    print(f"\n  Reports: {Path(output_dir).resolve()}")
    print(f"  JSON:    {out_path.resolve()}")
    print("=" * 55)

    return summary
