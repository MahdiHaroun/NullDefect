"""
pipelines/inference_pipeline.py

NullDefect — ZenML Inference Pipeline

Runs all three models on a single image and returns combined output.
Called by the FastAPI app for each /predict request.

Usage (direct):
    python pipelines/inference_pipeline.py --image path/to/image.png

Usage (via FastAPI):
    uvicorn src.serving.app:app --reload
    curl -X POST http://localhost:8000/predict -F "file=@image.png"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import yaml
from zenml import pipeline
from zenml.logger import get_logger

from steps.inference.inference_steps import (
    preprocess_step,
    classify_step,
    anomaly_step,
    segment_step,
    postprocess_step,
)

logger = get_logger(__name__)


# ── Pipeline ──────────────────────────────────────────────────────────────────

@pipeline(
    name="nulldefect_inference",
    enable_cache=False,   # inference is always fresh — never cache predictions
)
def inference_pipeline(
    image_path:      str,
    classifier_ckpt: str,
    segmenter_ckpt:  str,
    clf_threshold:   float = 0.5,
    seg_threshold:   float = 0.5,
    device:          str   = "cpu",
):
    """
    Single-image inference pipeline.

    Pipeline DAG:
      preprocess → classify ──────────────────────────────┐
                 → anomaly  ──────────────────────────────┤→ postprocess
                 → segment  ──────────────────────────────┘

    ZenML tracks:
      - image_path as input artifact
      - Each step's output as versioned artifact
      - Full lineage from image → prediction
    """
    # Step 1: Load image
    image = preprocess_step(image_path=image_path)

    # Step 2: Run all three models (independent — could run in parallel)
    clf_result = classify_step(
        image=image,
        classifier_ckpt=classifier_ckpt,
        threshold=clf_threshold,
        device=device,
    )

    ano_result = anomaly_step(
        image=image,
        classifier_ckpt=classifier_ckpt,
        threshold=clf_threshold,
        device=device,
    )

    seg_result = segment_step(
        image=image,
        segmenter_ckpt=segmenter_ckpt,
        threshold=seg_threshold,
        device=device,
    )

    # Step 3: Generate Grad-CAM + combine into final output
    output = postprocess_step(
        image=image,
        classification_result=clf_result,
        anomaly_result=ano_result,
        segmentation_result=seg_result,
        classifier_ckpt=classifier_ckpt,
        device=device,
    )

    return output


# ── Auto-detect checkpoints ───────────────────────────────────────────────────

def find_best_ckpt(directory: str) -> str:
    ckpt_dir = Path(directory)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    ckpts = [c for c in ckpt_dir.glob("*.ckpt") if "last" not in c.name]
    if not ckpts:
        ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")
    ckpts.sort(key=lambda p: p.name, reverse=True)
    return str(ckpts[0])


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run NullDefect inference on one image")
    parser.add_argument("--image",       required=True, help="Path to input image")
    parser.add_argument("--classifier-ckpt", default=None)
    parser.add_argument("--segmenter-ckpt",  default=None)
    parser.add_argument("--device",      default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--output",      default=None, help="Save result JSON to file")
    args = parser.parse_args()

    clf_ckpt = args.classifier_ckpt or find_best_ckpt("checkpoints/classifier")
    seg_ckpt = args.segmenter_ckpt  or find_best_ckpt("checkpoints/segmentation")

    print(f"\n  Image            : {args.image}")
    print(f"  Classifier ckpt  : {clf_ckpt}")
    print(f"  Segmenter ckpt   : {seg_ckpt}")
    print(f"  Device           : {args.device}\n")

    # Run ZenML pipeline
    inference_pipeline(
        image_path=args.image,
        classifier_ckpt=clf_ckpt,
        segmenter_ckpt=seg_ckpt,
        device=args.device,
    )

    print("\n  View results in ZenML dashboard: http://localhost:8080")


if __name__ == "__main__":
    main()
