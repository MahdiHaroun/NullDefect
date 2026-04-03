"""
src/data/split.py

Stratified train / val / test split — image level, not patch level.

Why image-level splitting matters:
  If you split at the patch level, patches from the same source image
  can appear in both train and test. The model memorizes the image
  rather than learning defect patterns. This is called data leakage.

  We split at the SOURCE IMAGE level: all patches from image_0001.png
  go exclusively to one split. This mirrors real production evaluation
  where the model will see completely new images at inference time.

Stratification:
  We stratify by (category, defect_type) so each split has a balanced
  representation of all 15 categories and all 73 defect types.

Output:
  data/processed/splits/
    train.csv  ← patch_path, category, defect_type, split, label, mask_path
    val.csv
    test.csv
  data/processed/metadata/dataset_metadata.csv  ← all splits combined

Usage:
    python src/data/split.py
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Metadata builder ──────────────────────────────────────────────────────────

def build_patch_records(patches_dir: Path, categories: list[str]) -> list[dict]:
    """
    Walks the patches directory and builds one record per patch file.

    Each record contains:
      - patch_path    : relative path to the image patch
      - mask_path     : relative path to the ground truth mask (or None)
      - category      : MVTec category (e.g. "bottle")
      - defect_type   : "good" or defect name (e.g. "broken_large")
      - label         : 0 = normal, 1 = defective
      - source_image  : original image stem (used for image-level splitting)
    """
    records = []

    for category in categories:
        cat_dir = patches_dir / category

        if not cat_dir.exists():
            print(f"  [WARN] Patches not found for category: {category} — run tile.py first")
            continue

        # ── Train (normal only) ───────────────────────────────────────────
        train_good_dir = cat_dir / "train" / "good"
        if train_good_dir.exists():
            for patch_path in sorted(train_good_dir.glob("*.png")):
                # Filename: {category}_train_good_{stem}_patch_{r}_{c}.png
                # Extract source image stem from filename
                parts = patch_path.stem.split("_patch_")
                source_image = parts[0] if len(parts) == 2 else patch_path.stem

                records.append({
                    "patch_path": str(patch_path),
                    "mask_path": None,
                    "category": category,
                    "defect_type": "good",
                    "label": 0,
                    "label_name": "normal",
                    "source_image": source_image,
                    "original_split": "train",  # MVTec's original split designation
                })

        # ── Test ──────────────────────────────────────────────────────────
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        if not test_dir.exists():
            continue

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            is_normal = defect_type == "good"
            label = 0 if is_normal else 1

            for patch_path in sorted(defect_dir.glob("*.png")):
                parts = patch_path.stem.split("_patch_")
                source_image = parts[0] if len(parts) == 2 else patch_path.stem

                # Find matching mask patch
                mask_path = None
                if not is_normal:
                    candidate = gt_dir / defect_type / patch_path.name
                    if candidate.exists():
                        mask_path = str(candidate)

                records.append({
                    "patch_path": str(patch_path),
                    "mask_path": mask_path,
                    "category": category,
                    "defect_type": defect_type,
                    "label": label,
                    "label_name": defect_type if not is_normal else "normal",
                    "source_image": source_image,
                    "original_split": "test",
                })

    return records


# ── Stratified image-level split ─────────────────────────────────────────────

def stratified_image_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    Splits at the source_image level.

    Strategy:
      1. Get unique source images with their (category, defect_type) strata.
      2. Split unique images into train/val/test (stratified).
      3. Assign the split label to all patches from each image.

    This guarantees no patch from a test image appears in train or val.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # One row per unique source image
    image_df = df.drop_duplicates("source_image")[
        ["source_image", "category", "defect_type", "label"]
    ].copy()

    # Stratification key: category + defect_type
    image_df["strata"] = image_df["category"] + "__" + image_df["defect_type"]

    # Some strata may have very few samples — merge rare ones into "other"
    strata_counts = image_df["strata"].value_counts()
    rare_strata = strata_counts[strata_counts < 3].index
    image_df["strata_safe"] = image_df["strata"].apply(
        lambda s: "rare__other" if s in rare_strata else s
    )

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_images, valtest_images = train_test_split(
        image_df,
        test_size=val_test_ratio,
        stratify=image_df["strata_safe"],
        random_state=seed,
    )

    # Second split: val vs test (within the val+test pool)
    relative_test_ratio = test_ratio / val_test_ratio
    val_images, test_images = train_test_split(
        valtest_images,
        test_size=relative_test_ratio,
        stratify=valtest_images["strata_safe"],
        random_state=seed,
    )

    # Build the split mapping: source_image → split name
    split_map = {}
    for img in train_images["source_image"]:
        split_map[img] = "train"
    for img in val_images["source_image"]:
        split_map[img] = "val"
    for img in test_images["source_image"]:
        split_map[img] = "test"

    df["split"] = df["source_image"].map(split_map)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stratified image-level data split")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patches_dir = Path(cfg["paths"]["patches_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    metadata_csv = Path(cfg["paths"]["metadata_csv"])
    categories = cfg["dataset"]["categories"]

    train_ratio = cfg["splits"]["train"]
    val_ratio = cfg["splits"]["val"]
    test_ratio = cfg["splits"]["test"]
    seed = cfg["splits"]["seed"]

    splits_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Building patch metadata & stratified splits")
    print("=" * 60)

    # Build records
    print("\n  Scanning patches directory...")
    records = build_patch_records(patches_dir, categories)

    if not records:
        print("  [ERROR] No patch records found. Run tile.py first.")
        return

    df = pd.DataFrame(records)
    print(f"  Total patches found: {len(df):,}")
    print(f"  Normal: {(df['label'] == 0).sum():,}  |  Defective: {(df['label'] == 1).sum():,}")

    # Split
    print("\n  Applying stratified image-level split...")
    df = stratified_image_split(df, train_ratio, val_ratio, test_ratio, seed)

    # Drop rows that couldn't be assigned (rare edge case with tiny strata)
    unassigned = df["split"].isna().sum()
    if unassigned > 0:
        print(f"  [WARN] {unassigned} patches could not be assigned to a split — dropping.")
        df = df.dropna(subset=["split"])

    # ── Save CSVs ─────────────────────────────────────────────────────────
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].copy()
        out_path = splits_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)

        normal_count = (split_df["label"] == 0).sum()
        defect_count = (split_df["label"] == 1).sum()
        print(
            f"  {split_name.upper():<6}: {len(split_df):>7,} patches  "
            f"(normal={normal_count:,}, defect={defect_count:,})"
        )

    # Save full metadata
    df.to_csv(metadata_csv, index=False)
    print(f"\n  [OK] Full metadata saved to: {metadata_csv}")

    # ── Category breakdown ────────────────────────────────────────────────
    print("\n  Category breakdown (train / val / test):")
    summary = df.groupby(["category", "split"]).size().unstack(fill_value=0)
    for cat in categories:
        if cat in summary.index:
            row = summary.loc[cat]
            print(
                f"    {cat:<16} "
                f"train={row.get('train', 0):<7} "
                f"val={row.get('val', 0):<7} "
                f"test={row.get('test', 0):<7}"
            )

    # ── Class imbalance report ────────────────────────────────────────────
    print("\n  Class imbalance per split:")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        n = (split_df["label"] == 0).sum()
        d = (split_df["label"] == 1).sum()
        ratio = n / d if d > 0 else float("inf")
        print(f"    {split_name.upper():<6}: normal/defect ratio = {ratio:.1f}:1")

    print("\n  Next step: python src/data/validate.py")


if __name__ == "__main__":
    main()
