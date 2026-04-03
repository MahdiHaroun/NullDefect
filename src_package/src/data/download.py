"""
src/data/download.py

MVTec AD download helper.

MVTec requires accepting a license before downloading, so the dataset
cannot be pulled programmatically without authentication. This script:
  1. Checks if the dataset already exists locally.
  2. If not, prints clear instructions for manual download.
  3. Verifies the folder structure after placement.

Usage:
    python src/data/download.py
"""

import os
import sys
from pathlib import Path

import yaml


# ── Load config ───────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Verification ──────────────────────────────────────────────────────────────

def verify_mvtec_structure(raw_dir: Path, categories: list[str]) -> bool:
    """
    Checks that each MVTec category folder exists with the expected subfolders.

    Expected structure per category:
        mvtec/
          bottle/
            train/good/         ← normal training images
            test/good/          ← normal test images
            test/broken_large/  ← defect type (name varies per category)
            ground_truth/       ← pixel-level binary masks
    """
    all_ok = True
    missing = []

    for category in categories:
        cat_path = raw_dir / category
        train_good = cat_path / "train" / "good"
        test_dir = cat_path / "test"
        gt_dir = cat_path / "ground_truth"

        if not cat_path.exists():
            missing.append(str(cat_path))
            all_ok = False
            continue

        if not train_good.exists():
            print(f"  [WARN] {category}: missing train/good/")
            all_ok = False

        if not test_dir.exists():
            print(f"  [WARN] {category}: missing test/")
            all_ok = False

        if not gt_dir.exists():
            print(f"  [WARN] {category}: missing ground_truth/")
            all_ok = False

    if missing:
        print(f"\n  Missing categories: {missing}")

    return all_ok


def count_images(raw_dir: Path, categories: list[str]) -> dict:
    """Count images per category and split for a quick sanity check."""
    stats = {}
    for category in categories:
        cat_path = raw_dir / category
        if not cat_path.exists():
            continue

        train_count = len(list((cat_path / "train" / "good").glob("*.png")))

        test_normal = len(list((cat_path / "test" / "good").glob("*.png")))
        test_defect = sum(
            len(list(d.glob("*.png")))
            for d in (cat_path / "test").iterdir()
            if d.is_dir() and d.name != "good"
        )

        defect_types = [
            d.name for d in (cat_path / "test").iterdir()
            if d.is_dir() and d.name != "good"
        ]

        stats[category] = {
            "train_normal": train_count,
            "test_normal": test_normal,
            "test_defect": test_defect,
            "defect_types": defect_types,
        }
    return stats


# ── Instructions ──────────────────────────────────────────────────────────────

DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║              MVTec AD — Manual Download Required                 ║
╚══════════════════════════════════════════════════════════════════╝

MVTec AD requires accepting a license agreement before downloading.

Step 1: Go to https://www.mvtec.com/company/research/datasets/mvtec-ad
Step 2: Click "Download Dataset" and accept the license.
Step 3: Download the full dataset (~4.9 GB): mvtec_anomaly_detection.tar.xz

Alternative (faster): Kaggle mirror
  https://www.kaggle.com/datasets/ipythonx/mvtec-ad
  → kaggle datasets download -d ipythonx/mvtec-ad

Step 4: Extract to:  data/raw/mvtec/
        Expected:    data/raw/mvtec/bottle/
                     data/raw/mvtec/cable/
                     ... (15 categories)

Step 5: Re-run this script to verify the structure.
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_dir"])
    categories = cfg["dataset"]["categories"]

    print("=" * 60)
    print("  MVTec AD — Dataset Verification")
    print("=" * 60)

    # Check if dataset exists at all
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(DOWNLOAD_INSTRUCTIONS)
        sys.exit(1)

    print(f"\n  Found dataset at: {raw_dir.resolve()}")
    print(f"  Verifying {len(categories)} categories...\n")

    ok = verify_mvtec_structure(raw_dir, categories)

    if not ok:
        print("\n  [ERROR] Dataset structure is incomplete.")
        print(DOWNLOAD_INSTRUCTIONS)
        sys.exit(1)

    # Count images and print summary table
    stats = count_images(raw_dir, categories)
    total_train = sum(s["train_normal"] for s in stats.values())
    total_test_normal = sum(s["test_normal"] for s in stats.values())
    total_test_defect = sum(s["test_defect"] for s in stats.values())

    print(f"  {'CATEGORY':<16} {'TRAIN(N)':<12} {'TEST(N)':<10} {'TEST(D)':<10} DEFECT TYPES")
    print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*10} {'-'*30}")
    for cat, s in stats.items():
        print(
            f"  {cat:<16} {s['train_normal']:<12} {s['test_normal']:<10} "
            f"{s['test_defect']:<10} {', '.join(s['defect_types'][:3])}{'...' if len(s['defect_types']) > 3 else ''}"
        )

    print(f"\n  {'TOTAL':<16} {total_train:<12} {total_test_normal:<10} {total_test_defect:<10}")
    print(f"\n  [OK] Dataset verified. {len(stats)} / {len(categories)} categories found.")
    print(f"  Total defect types across all categories: "
          f"{sum(len(s['defect_types']) for s in stats.values())}")
    print("\n  Ready for next step: python src/data/tile.py")


if __name__ == "__main__":
    main()
