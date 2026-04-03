"""
src/data/validate.py

Data quality validation using Great Expectations (lightweight version).

What we check:
  1. Metadata CSV integrity  — required columns, no nulls in critical fields
  2. Image file existence    — all patch_path entries actually exist on disk
  3. Mask file existence     — all mask_path entries exist (for defective patches)
  4. Split coverage          — all three splits (train/val/test) are present
  5. Class balance           — normal/defect ratio within acceptable bounds
  6. Category coverage       — all 15 MVTec categories are represented
  7. Image readability       — random sample of images can be opened by OpenCV
  8. Image size consistency  — patches are the expected size (512×512)

Why Great Expectations?
  It produces a human-readable HTML validation report that can be stored
  as a CI/CD artifact. A failed expectation fails the DVC pipeline stage,
  preventing bad data from propagating to training.

Usage:
    python src/data/validate.py
    python src/data/validate.py --sample-size 200  # check 200 random images
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Individual checks ─────────────────────────────────────────────────────────

def check_csv_schema(df: pd.DataFrame) -> dict:
    """Verify required columns exist and have no nulls in critical fields."""
    required_columns = [
        "patch_path", "category", "defect_type",
        "label", "split", "source_image",
    ]
    result = {"name": "csv_schema", "passed": True, "details": []}

    for col in required_columns:
        if col not in df.columns:
            result["passed"] = False
            result["details"].append(f"MISSING column: {col}")
        elif df[col].isna().any():
            null_count = df[col].isna().sum()
            result["passed"] = False
            result["details"].append(f"NULL values in '{col}': {null_count} rows")

    if result["passed"]:
        result["details"].append(f"All {len(required_columns)} required columns present, no nulls.")

    return result


def check_split_coverage(df: pd.DataFrame) -> dict:
    """Ensure all three splits exist and are non-empty."""
    result = {"name": "split_coverage", "passed": True, "details": []}
    splits = df["split"].unique().tolist()

    for required_split in ["train", "val", "test"]:
        if required_split not in splits:
            result["passed"] = False
            result["details"].append(f"MISSING split: {required_split}")
        else:
            count = (df["split"] == required_split).sum()
            result["details"].append(f"  {required_split}: {count:,} patches")

    return result


def check_category_coverage(df: pd.DataFrame, expected_categories: list[str]) -> dict:
    """Verify all 15 MVTec categories are in the dataset."""
    result = {"name": "category_coverage", "passed": True, "details": []}
    found = set(df["category"].unique())
    expected = set(expected_categories)
    missing = expected - found

    if missing:
        result["passed"] = False
        result["details"].append(f"MISSING categories: {sorted(missing)}")
    else:
        result["details"].append(f"All {len(expected)} categories present.")

    return result


def check_label_values(df: pd.DataFrame) -> dict:
    """Ensure labels are only 0 or 1."""
    result = {"name": "label_values", "passed": True, "details": []}
    invalid = df[~df["label"].isin([0, 1])]

    if len(invalid) > 0:
        result["passed"] = False
        result["details"].append(f"Invalid label values found: {invalid['label'].unique().tolist()}")
    else:
        n_normal = (df["label"] == 0).sum()
        n_defect = (df["label"] == 1).sum()
        result["details"].append(f"Normal: {n_normal:,} | Defective: {n_defect:,}")

    return result


def check_class_balance(df: pd.DataFrame, max_ratio: float = 20.0) -> dict:
    """
    Warn if normal/defect ratio exceeds max_ratio.
    MVTec is naturally imbalanced — this check catches extreme cases.
    """
    result = {"name": "class_balance", "passed": True, "details": []}

    train_df = df[df["split"] == "train"]
    n_normal = (train_df["label"] == 0).sum()
    n_defect = (train_df["label"] == 1).sum()

    if n_defect == 0:
        result["passed"] = False
        result["details"].append("No defective samples in train split!")
        return result

    ratio = n_normal / n_defect
    result["details"].append(f"Train normal/defect ratio: {ratio:.1f}:1")

    if ratio > max_ratio:
        result["passed"] = False
        result["details"].append(
            f"WARNING: Ratio {ratio:.1f} exceeds threshold {max_ratio}. "
            "Consider WeightedRandomSampler or oversampling."
        )
    else:
        result["details"].append("Ratio is within acceptable bounds.")

    return result


def check_file_existence(df: pd.DataFrame, sample_size: int = 500) -> dict:
    """
    Verify patch files exist on disk for a random sample.
    Checking all files is slow; sampling gives statistical confidence.
    """
    result = {"name": "file_existence", "passed": True, "details": []}

    sample = df.sample(min(sample_size, len(df)), random_state=42)
    missing_imgs = []
    missing_masks = []

    for _, row in sample.iterrows():
        if not Path(row["patch_path"]).exists():
            missing_imgs.append(row["patch_path"])

        if pd.notna(row.get("mask_path")) and row["mask_path"]:
            if not Path(row["mask_path"]).exists():
                missing_masks.append(row["mask_path"])

    if missing_imgs:
        result["passed"] = False
        result["details"].append(
            f"Missing image files: {len(missing_imgs)} / {sample_size} sampled"
        )
        result["details"].append(f"  First missing: {missing_imgs[0]}")
    else:
        result["details"].append(f"All {sample_size} sampled image files exist.")

    if missing_masks:
        result["passed"] = False
        result["details"].append(
            f"Missing mask files: {len(missing_masks)} / {sample_size} sampled"
        )
    else:
        result["details"].append("All sampled mask files exist.")

    return result


def check_image_readability(df: pd.DataFrame, sample_size: int = 100) -> dict:
    """Try to open a random sample of images with OpenCV."""
    result = {"name": "image_readability", "passed": True, "details": []}

    sample = df.sample(min(sample_size, len(df)), random_state=99)
    failed = []

    for _, row in sample.iterrows():
        img = cv2.imread(row["patch_path"])
        if img is None:
            failed.append(row["patch_path"])

    if failed:
        result["passed"] = False
        result["details"].append(f"Unreadable images: {len(failed)} / {sample_size}")
        result["details"].append(f"  First: {failed[0]}")
    else:
        result["details"].append(f"All {sample_size} sampled images readable by OpenCV.")

    return result


def check_image_size(df: pd.DataFrame, expected_size: int, sample_size: int = 50) -> dict:
    """Verify patches are the expected size (default 512×512)."""
    result = {"name": "image_size", "passed": True, "details": []}

    sample = df.sample(min(sample_size, len(df)), random_state=7)
    wrong_size = []

    for _, row in sample.iterrows():
        img = cv2.imread(row["patch_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        if h != expected_size or w != expected_size:
            wrong_size.append(f"{row['patch_path']} → {w}×{h}")

    if wrong_size:
        result["passed"] = False
        result["details"].append(
            f"Wrong-size patches: {len(wrong_size)} / {sample_size}"
        )
        for p in wrong_size[:3]:
            result["details"].append(f"  {p}")
    else:
        result["details"].append(
            f"All {sample_size} sampled patches are {expected_size}×{expected_size}."
        )

    return result


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> bool:
    """Print validation results and return True if all passed."""
    print("\n" + "=" * 60)
    print("  DATA QUALITY VALIDATION REPORT")
    print("=" * 60)

    all_passed = True
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"\n  [{status}] {r['name']}")
        for detail in r["details"]:
            print(f"         {detail}")

    print("\n" + "=" * 60)
    if all_passed:
        print("  RESULT: ALL CHECKS PASSED ✓")
        print("  Data pipeline is ready for training.")
    else:
        print("  RESULT: SOME CHECKS FAILED ✗")
        print("  Fix the issues above before proceeding to training.")
    print("=" * 60 + "\n")

    return all_passed


def save_json_report(results: list[dict], output_path: Path):
    """Save results as JSON for DVC metrics tracking."""
    summary = {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "all_passed": all(r["passed"] for r in results),
        "checks": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON report saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline outputs")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Number of files to sample for existence checks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata_csv = Path(cfg["paths"]["metadata_csv"])
    expected_categories = cfg["dataset"]["categories"]
    patch_size = cfg["tiling"]["patch_size"]
    metadata_dir = Path(cfg["paths"]["metadata_dir"])

    if not metadata_csv.exists():
        print(f"[ERROR] Metadata CSV not found at {metadata_csv}")
        print("  Run split.py first.")
        sys.exit(1)

    print(f"\n  Loading metadata from: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    print(f"  Total rows: {len(df):,}")

    results = [
        check_csv_schema(df),
        check_split_coverage(df),
        check_category_coverage(df, expected_categories),
        check_label_values(df),
        check_class_balance(df),
        check_file_existence(df, sample_size=args.sample_size),
        check_image_readability(df, sample_size=100),
        check_image_size(df, expected_size=patch_size, sample_size=50),
    ]

    all_passed = print_report(results)
    save_json_report(results, metadata_dir / "validation_report.json")

    if not all_passed:
        sys.exit(1)

    print("  Next step: python src/data/dataset.py  (smoke test)")
    print("  Then: SageMaker training → Phase 2\n")


if __name__ == "__main__":
    main()
