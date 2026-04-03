"""
src/evaluation/evaluate.py

Master evaluation entry point — runs all five portfolio outputs:
  1. Grad-CAM++ heatmaps (defect localization visualization)
  2. Per-category AUROC bar chart (all 15 MVTec categories)
  3. Calibration curves (reliability diagram + confidence histogram)
  4. Evidently HTML reports (classification, data quality, segmentation)
  5. Benchmark comparison table (your PatchCore vs published)

Also logs all metrics and artifacts to MLflow.

Usage:
    # Evaluate all three models
    python src/evaluation/evaluate.py

    # Evaluate only the classifier
    python src/evaluation/evaluate.py --model classifier

    # Skip Grad-CAM (slow) — useful for quick metric check
    python src/evaluation/evaluate.py --skip-gradcam

    # Load checkpoints from S3
    python src/evaluation/evaluate.py --from-s3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import mlflow
import torch
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/evaluate.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_data_config(path: str = "configs/data.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_classifier(ckpt_path: str, train_cfg_path: str, device: str):
    """Load EfficientNet-B4 from Lightning checkpoint."""
    from models.classifier import DefectClassifier
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    model = DefectClassifier.load_from_checkpoint(ckpt_path, cfg=train_cfg)
    return model.to(device).eval()


def load_segmenter(ckpt_path: str, train_cfg_path: str, device: str):
    """Load U-Net from Lightning checkpoint."""
    from models.segmentation import DefectSegmenter
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    model = DefectSegmenter.load_from_checkpoint(ckpt_path, cfg=train_cfg)
    return model.to(device).eval()


# ── Dataloader builders ───────────────────────────────────────────────────────

def build_classifier_test_loader(metadata_csv: str, cfg: dict, data_cfg: dict):
    from data.transforms import get_classifier_transforms
    from data.dataset import DefectClassificationDataset
    from torch.utils.data import DataLoader

    transforms = get_classifier_transforms(cfg["gradcam"]["image_size"])
    ds = DefectClassificationDataset(
        csv_path=metadata_csv,
        split="test",
        transform=transforms["test"],
    )
    return DataLoader(
        ds,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["evaluation"]["num_workers"],
    ), transforms


def build_segmentation_test_loader(metadata_csv: str, cfg: dict, data_cfg: dict):
    from data.transforms import get_segmentation_transforms
    from data.dataset import DefectSegmentationDataset
    from torch.utils.data import DataLoader

    transforms = get_segmentation_transforms(cfg["gradcam"]["image_size"])
    ds = DefectSegmentationDataset(
        csv_path=metadata_csv,
        split="test",
        transform=transforms["test"],
        defect_only=True,
    )
    return DataLoader(
        ds,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["evaluation"]["num_workers"],
    )


# ── Evaluation runners ────────────────────────────────────────────────────────

def evaluate_classifier(cfg: dict, data_cfg: dict, skip_gradcam: bool) -> dict:
    """Full evaluation pipeline for EfficientNet-B4."""
    from evaluation.metrics import ClassifierMetrics, compute_per_category_auroc
    from evaluation.calibration import (
        plot_calibration_curve,
        plot_per_category_auroc,
        plot_roc_curve,
    )
    from evaluation.report import generate_classification_report, generate_data_quality_report
    from evaluation.gradcam import generate_gradcam_for_dataset

    print("\n" + "=" * 55)
    print("  EVALUATING: EfficientNet-B4 Classifier")
    print("=" * 55)

    device = cfg["evaluation"]["device"] if torch.cuda.is_available() else "cpu"
    ckpt   = cfg["checkpoints"]["classifier"]
    metadata_csv = data_cfg["paths"]["metadata_csv"]

    model = load_classifier(ckpt, "configs/train_classifier.yaml", device)
    test_loader, transforms = build_classifier_test_loader(metadata_csv, cfg, data_cfg)

    # ── Core metrics ──────────────────────────────────────────────────────
    metrics_runner = ClassifierMetrics(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=cfg["evaluation"]["classifier_threshold"],
    )
    results = metrics_runner.compute()

    # ── Per-category AUROC ────────────────────────────────────────────────
    print("\n  Per-category AUROC breakdown:")
    per_cat = compute_per_category_auroc(
        model=model,
        metadata_csv_path=metadata_csv,
        transforms=transforms,
        cfg=cfg,
        device=device,
    )
    results["per_category_auroc"] = per_cat

    # ── Plots ─────────────────────────────────────────────────────────────
    plots_dir = cfg["output"]["plots_dir"]
    print("\n  Generating plots...")

    plot_calibration_curve(
        probs=results["_probs"],
        labels=results["_labels"],
        output_dir=plots_dir,
        model_name="EfficientNet-B4",
        n_bins=cfg["calibration"]["n_bins"],
    )
    plot_per_category_auroc(
        per_cat_auroc=per_cat,
        output_dir=plots_dir,
        model_name="Classifier",
    )
    plot_roc_curve(
        probs=results["_probs"],
        labels=results["_labels"],
        output_dir=plots_dir,
        model_name="Classifier",
    )

    # ── Evidently reports ─────────────────────────────────────────────────
    print("\n  Generating Evidently reports...")
    evidently_dir = cfg["output"]["evidently_dir"]

    generate_classification_report(
        probs=results["_probs"],
        labels=results["_labels"],
        output_dir=evidently_dir,
        model_name="EfficientNet-B4",
        threshold=cfg["evaluation"]["classifier_threshold"],
    )
    generate_data_quality_report(
        metadata_csv_path=metadata_csv,
        output_dir=evidently_dir,
    )

    # ── Grad-CAM++ ────────────────────────────────────────────────────────
    if not skip_gradcam:
        print("\n  Generating Grad-CAM++ heatmaps...")
        generate_gradcam_for_dataset(
            model=model,
            metadata_csv_path=metadata_csv,
            raw_data_dir=data_cfg["paths"]["raw_dir"],
            output_dir=cfg["output"]["gradcam_dir"],
            cfg=cfg,
            device=device,
        )
    else:
        print("\n  [SKIP] Grad-CAM++ (--skip-gradcam flag set)")

    # Remove internal arrays before saving/logging (too large for JSON)
    results_clean = {k: v for k, v in results.items() if not k.startswith("_")}
    return results_clean


def evaluate_anomaly(cfg: dict, data_cfg: dict) -> dict:
    """Full evaluation pipeline for PatchCore."""
    from evaluation.metrics import AnomalyMetrics
    from evaluation.calibration import plot_per_category_auroc, plot_benchmark_comparison

    print("\n" + "=" * 55)
    print("  EVALUATING: PatchCore Anomaly Detector")
    print("=" * 55)

    results_json = Path(cfg["checkpoints"]["anomaly_dir"]) / "results_summary.json"

    if not results_json.exists():
        print(f"  [WARN] Anomaly results JSON not found at {results_json}")
        print("  Run train_anomaly.py first.")
        return {}

    metrics = AnomalyMetrics(str(results_json))
    results = metrics.compute()

    plots_dir = cfg["output"]["plots_dir"]

    # Per-category AUROC chart with published comparison
    per_cat_simple = {
        cat: vals["your_image_auroc"]
        for cat, vals in results["per_category"].items()
    }
    per_cat_simple["_mean"] = results["mean_image_auroc"]

    plot_per_category_auroc(
        per_cat_auroc=per_cat_simple,
        output_dir=plots_dir,
        model_name="PatchCore",
        published={
            cat: vals["published_image_auroc"]
            for cat, vals in results["per_category"].items()
        },
    )

    # Benchmark comparison table
    plot_benchmark_comparison(
        anomaly_results=results,
        output_dir=plots_dir,
    )

    return results


def evaluate_segmentation(cfg: dict, data_cfg: dict) -> dict:
    """Full evaluation pipeline for U-Net segmenter."""
    from evaluation.metrics import SegmentationMetrics
    from evaluation.calibration import plot_calibration_curve, plot_roc_curve
    from evaluation.report import generate_segmentation_report

    print("\n" + "=" * 55)
    print("  EVALUATING: U-Net Segmentation Model")
    print("=" * 55)

    device = cfg["evaluation"]["device"] if torch.cuda.is_available() else "cpu"
    ckpt   = cfg["checkpoints"]["segmentation"]
    metadata_csv = data_cfg["paths"]["metadata_csv"]

    model = load_segmenter(ckpt, "configs/train_segmentation.yaml", device)
    test_loader = build_segmentation_test_loader(metadata_csv, cfg, data_cfg)

    metrics_runner = SegmentationMetrics(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=cfg["evaluation"]["segmentation_threshold"],
    )
    results = metrics_runner.compute()

    plots_dir    = cfg["output"]["plots_dir"]
    evidently_dir = cfg["output"]["evidently_dir"]

    plot_calibration_curve(
        probs=results["_probs"],
        labels=results["_labels"],
        output_dir=plots_dir,
        model_name="U-Net Segmenter (Pixel-level)",
    )
    plot_roc_curve(
        probs=results["_probs"],
        labels=results["_labels"],
        output_dir=plots_dir,
        model_name="Segmenter Pixel",
    )
    generate_segmentation_report(
        pixel_probs=results["_probs"],
        pixel_labels=results["_labels"],
        output_dir=evidently_dir,
        threshold=cfg["evaluation"]["segmentation_threshold"],
    )

    return {k: v for k, v in results.items() if not k.startswith("_")}


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    classifier_results: dict,
    anomaly_results: dict,
    segmentation_results: dict,
    cfg: dict,
):
    """Log all metrics and artifact paths to MLflow."""
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "defect_evaluation"))

    with mlflow.start_run(run_name="full_evaluation"):
        # Classifier metrics
        if classifier_results:
            mlflow.log_metrics({
                "clf_auroc":         classifier_results.get("auroc", 0),
                "clf_f1":            classifier_results.get("f1", 0),
                "clf_precision":     classifier_results.get("precision", 0),
                "clf_recall":        classifier_results.get("recall", 0),
                "clf_mean_cat_auroc": classifier_results.get(
                    "per_category_auroc", {}).get("_mean", 0),
            })

        # Anomaly metrics
        if anomaly_results:
            mlflow.log_metrics({
                "anomaly_mean_image_auroc": anomaly_results.get("mean_image_auroc", 0),
                "anomaly_mean_pixel_auroc": anomaly_results.get("mean_pixel_auroc", 0),
                "anomaly_delta_vs_published": anomaly_results.get("mean_delta", 0),
            })

        # Segmentation metrics
        if segmentation_results:
            mlflow.log_metrics({
                "seg_dice":        segmentation_results.get("dice", 0),
                "seg_iou":         segmentation_results.get("iou", 0),
                "seg_pixel_auroc": segmentation_results.get("pixel_auroc", 0),
            })

        # Log all artifact directories
        if mlflow_cfg.get("log_artifacts", True):
            for artifact_dir in [
                cfg["output"]["plots_dir"],
                cfg["output"]["evidently_dir"],
                cfg["output"]["gradcam_dir"],
            ]:
                if Path(artifact_dir).exists():
                    mlflow.log_artifacts(artifact_dir)

        print("\n  All metrics and artifacts logged to MLflow.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--model", choices=["classifier", "anomaly", "segmentation", "all"],
                        default="all", help="Which model to evaluate")
    parser.add_argument("--config",       default="configs/evaluate.yaml")
    parser.add_argument("--skip-gradcam", action="store_true",
                        help="Skip Grad-CAM++ generation (faster)")
    parser.add_argument("--from-s3",      action="store_true",
                        help="Download checkpoints from S3 if not found locally")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = load_data_config()

    # Create output directories
    for d in [cfg["output"]["reports_dir"], cfg["output"]["gradcam_dir"],
              cfg["output"]["plots_dir"], cfg["output"]["evidently_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    classifier_results   = {}
    anomaly_results      = {}
    segmentation_results = {}

    if args.model in ("classifier", "all"):
        classifier_results = evaluate_classifier(cfg, data_cfg, args.skip_gradcam)

    if args.model in ("anomaly", "all"):
        anomaly_results = evaluate_anomaly(cfg, data_cfg)

    if args.model in ("segmentation", "all"):
        segmentation_results = evaluate_segmentation(cfg, data_cfg)

    # ── Save combined results JSON ─────────────────────────────────────────
    combined = {
        "classifier":   classifier_results,
        "anomaly":      anomaly_results,
        "segmentation": segmentation_results,
    }
    results_path = Path(cfg["output"]["reports_dir"]) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined results saved: {results_path}")

    # ── MLflow logging ─────────────────────────────────────────────────────
    log_to_mlflow(classifier_results, anomaly_results, segmentation_results, cfg)

    elapsed = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"  Evaluation complete in {elapsed/60:.1f} minutes")
    print(f"  Reports directory: {Path(cfg['output']['reports_dir']).resolve()}")
    print(f"{'='*55}\n")

    # ── Print summary table ────────────────────────────────────────────────
    print("  SUMMARY")
    print(f"  {'─'*40}")
    if classifier_results:
        print(f"  Classifier  AUROC : {classifier_results.get('auroc', 'N/A')}")
        print(f"  Classifier  F1    : {classifier_results.get('f1', 'N/A')}")
    if anomaly_results:
        print(f"  PatchCore   AUROC : {anomaly_results.get('mean_image_auroc', 'N/A')}")
        print(f"  PatchCore   Δpub  : {anomaly_results.get('mean_delta', 'N/A'):+.4f}" if anomaly_results.get("mean_delta") is not None else "")
    if segmentation_results:
        print(f"  Segmenter   Dice  : {segmentation_results.get('dice', 'N/A')}")
        print(f"  Segmenter   IoU   : {segmentation_results.get('iou', 'N/A')}")
    print(f"  {'─'*40}")
    print(f"\n  Next step: Phase 4 — Quantization\n")


if __name__ == "__main__":
    main()
