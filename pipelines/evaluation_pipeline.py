"""
pipelines/evaluation_pipeline.py

NullDefect — Phase 3 Evaluation Pipeline (ZenML)

This is the main ZenML pipeline that orchestrates all evaluation steps.

What ZenML adds on top of plain Python:
  - Each step's outputs are versioned artifacts (tracked in the dashboard)
  - Full lineage: you can trace any metric back to the exact data + model that produced it
  - Caching: if nothing changed, ZenML skips the step and reuses the cached output
  - The dashboard shows a DAG (directed acyclic graph) of the pipeline
  - Re-runs are reproducible — same inputs always produce same outputs

Pipeline DAG:
                    ┌─────────────────────┐
                    │   classifier_metrics │
                    └──────────┬──────────┘
                               │
  ┌──────────────────┐         │        ┌─────────────────────┐
  │  anomaly_metrics │─────────┼────────│  segmentation_metrics│
  └──────────────────┘         │        └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   gradcam_step       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  calibration_plots   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  evidently_reports   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    summary_step      │
                    └─────────────────────┘

Usage:
    # First time setup
    python zenml_stack/setup_stack.py

    # Start ZenML dashboard
    zenml up

    # Run the pipeline
    python pipelines/evaluation_pipeline.py

    # Run with custom checkpoints
    python pipelines/evaluation_pipeline.py \
        --classifier-ckpt checkpoints/classifier-epoch=16-val_auroc=0.9226.ckpt \
        --segmentation-ckpt checkpoints/segmenter-epoch=39-val_dice=0.8553.ckpt

    # View run in dashboard
    # http://localhost:8080
"""

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import yaml
from zenml import pipeline
from zenml.logger import get_logger

from steps.evaluation.compute_metrics import (
    classifier_metrics_step,
    anomaly_metrics_step,
    segmentation_metrics_step,
)
from steps.evaluation.gradcam_step import gradcam_step
from steps.evaluation.calibration_step import calibration_plots_step
from steps.evaluation.evidently_step import evidently_reports_step
from steps.evaluation.summary_step import summary_step

logger = get_logger(__name__)


# ── Pipeline definition ───────────────────────────────────────────────────────

@pipeline(
    name="nulldefect_evaluation",
    enable_cache=True,   # ZenML will skip steps whose inputs haven't changed
)
def evaluation_pipeline(
    classifier_checkpoint: str,
    segmentation_checkpoint: str,
    anomaly_results_json: str,
    metadata_csv: str,
    device: str = "cuda",
    output_dir: str = "reports",
    gradcam_samples: int = 4,
):
    """
    Full Phase 3 evaluation pipeline for NullDefect.

    Parameters are passed in at runtime — no hardcoded paths.
    ZenML tracks all parameters + outputs as versioned artifacts.
    """

    # ── Step 1: Compute metrics for all three models ───────────────────────
    # These three steps can run in parallel (no data dependency between them)
    clf_metrics = classifier_metrics_step(
        checkpoint_path=classifier_checkpoint,
        metadata_csv=metadata_csv,
        device=device,
    )

    ano_metrics = anomaly_metrics_step(
        results_json_path=anomaly_results_json,
    )

    seg_metrics = segmentation_metrics_step(
        checkpoint_path=segmentation_checkpoint,
        metadata_csv=metadata_csv,
        device=device,
    )

    # ── Step 2: Grad-CAM++ heatmaps ────────────────────────────────────────
    gradcam_outputs = gradcam_step(
        checkpoint_path=classifier_checkpoint,
        metadata_csv=metadata_csv,
        output_dir=f"{output_dir}/gradcam",
        samples_per_defect_type=gradcam_samples,
        device=device,
    )

    # ── Step 3: Calibration curves + AUROC charts ──────────────────────────
    plot_paths = calibration_plots_step(
        classifier_metrics=clf_metrics,
        anomaly_metrics=ano_metrics,
        output_dir=f"{output_dir}/plots",
    )

    # ── Step 4: Evidently HTML reports ─────────────────────────────────────
    report_paths = evidently_reports_step(
        classifier_metrics=clf_metrics,
        segmentation_metrics=seg_metrics,
        metadata_csv=metadata_csv,
        output_dir=f"{output_dir}/evidently",
    )

    # ── Step 5: Final summary ──────────────────────────────────────────────
    summary = summary_step(
        classifier_metrics=clf_metrics,
        anomaly_metrics=ano_metrics,
        segmentation_metrics=seg_metrics,
        plot_paths=plot_paths,
        report_paths=report_paths,
        output_dir=output_dir,
    )

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run NullDefect Phase 3 evaluation")
    parser.add_argument(
        "--classifier-ckpt",
        default=None,
        help="Path to classifier checkpoint (auto-detected if omitted)",
    )
    parser.add_argument(
        "--segmentation-ckpt",
        default=None,
        help="Path to segmentation checkpoint (auto-detected if omitted)",
    )
    parser.add_argument(
        "--anomaly-json",
        default="checkpoints/anomaly/results_summary.json",
        help="Path to PatchCore results JSON",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Path to dataset metadata CSV (auto-detects from configs/data.yaml)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
    )
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=4,
        help="Grad-CAM++ images per defect type",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable ZenML caching — force re-run all steps",
    )
    args = parser.parse_args()

    # ── Auto-detect checkpoints ────────────────────────────────────────────
    def _extract_metric_score(ckpt_name: str) -> float:
        """Extract metric score from names like val_auroc=0.9226 or val_dice=0.8553."""
        match = re.search(
            r"(?:val[_-]?(?:auroc|dice|iou|f1)|score|auc)=([0-9]*\.?[0-9]+)",
            ckpt_name.lower(),
        )
        return float(match.group(1)) if match else 0.0

    def _resolve_checkpoint(
        model_type: str,
        cli_override: str | None,
        config_candidate: str | None,
    ) -> str:
        """Resolve checkpoint from CLI, config, or recursive scan under checkpoints/."""
        if cli_override:
            override_path = Path(cli_override)
            if not override_path.exists():
                raise FileNotFoundError(f"{model_type} checkpoint not found: {override_path}")
            return str(override_path)

        candidates: list[Path] = []

        # 1) Try explicit checkpoint path from evaluate config
        if config_candidate:
            cfg_path = Path(config_candidate)
            if cfg_path.exists() and cfg_path.suffix == ".ckpt":
                candidates.append(cfg_path)
            else:
                logger.warning(
                    "Configured %s checkpoint does not exist or is not a .ckpt: %s",
                    model_type,
                    config_candidate,
                )

        # 2) Scan checkpoints/ recursively (supports flat and nested layouts)
        checkpoints_root = Path("checkpoints")
        if checkpoints_root.exists():
            all_ckpts = list(checkpoints_root.rglob("*.ckpt"))
            key_terms = {
                "classifier": ["classifier"],
                "segmentation": ["segmenter", "segmentation", "segment"],
            }[model_type]

            filtered = [
                p
                for p in all_ckpts
                if any(
                    term in p.name.lower() or term in str(p.parent).lower()
                    for term in key_terms
                )
            ]
            candidates.extend(filtered or all_ckpts)

        # De-duplicate while preserving order
        unique: list[Path] = []
        seen = set()
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(c)

        if not unique:
            raise FileNotFoundError(
                f"No {model_type} checkpoint found. "
                f"Pass --{model_type}-ckpt explicitly or add .ckpt files under checkpoints/."
            )

        unique.sort(
            key=lambda p: (
                _extract_metric_score(p.name),
                "last" not in p.name.lower(),
                p.name.lower(),
            ),
            reverse=True,
        )
        return str(unique[0])

    eval_cfg_path = Path("configs/evaluate.yaml")
    eval_cfg = {}
    if eval_cfg_path.exists():
        with open(eval_cfg_path) as f:
            eval_cfg = yaml.safe_load(f) or {}

    checkpoints_cfg = eval_cfg.get("checkpoints", {}) if isinstance(eval_cfg, dict) else {}

    clf_ckpt = _resolve_checkpoint(
        model_type="classifier",
        cli_override=args.classifier_ckpt,
        config_candidate=checkpoints_cfg.get("classifier"),
    )
    seg_ckpt = _resolve_checkpoint(
        model_type="segmentation",
        cli_override=args.segmentation_ckpt,
        config_candidate=checkpoints_cfg.get("segmentation"),
    )

    # ── Auto-detect metadata CSV ───────────────────────────────────────────
    if args.metadata_csv:
        metadata_csv = args.metadata_csv
    else:
        with open("configs/data.yaml") as f:
            data_cfg = yaml.safe_load(f)
        metadata_csv = data_cfg["paths"]["metadata_csv"]

    print("\n" + "=" * 55)
    print("  NullDefect — Phase 3 Evaluation Pipeline")
    print("=" * 55)
    print(f"  Classifier ckpt  : {clf_ckpt}")
    print(f"  Segmenter ckpt   : {seg_ckpt}")
    print(f"  Anomaly JSON     : {args.anomaly_json}")
    print(f"  Metadata CSV     : {metadata_csv}")
    print(f"  Device           : {args.device}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  Cache            : {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 55 + "\n")

    # ── Disable cache if requested ─────────────────────────────────────────
    if args.no_cache:
        os.environ["ZENML_DISABLE_STEP_CACHE"] = "1"

    # ── Run pipeline ───────────────────────────────────────────────────────
    evaluation_pipeline(
        classifier_checkpoint=clf_ckpt,
        segmentation_checkpoint=seg_ckpt,
        anomaly_results_json=args.anomaly_json,
        metadata_csv=metadata_csv,
        device=args.device,
        output_dir=args.output_dir,
        gradcam_samples=args.gradcam_samples,
    )

    print("\n  Pipeline complete.")
    print("  View results in the ZenML dashboard: http://localhost:8080")
    print(f"  Reports: {Path(args.output_dir).resolve()}\n")


if __name__ == "__main__":
    main()
