"""
src/data/tile.py

Patch extractor — tiles high-resolution MVTec images into 512×512 patches.

Why tiling?
  MVTec images are up to 1024×1024. Training directly on full resolution is
  GPU memory-intensive. Tiling into 512×512 patches with overlap gives us:
    - More training samples per image
    - Consistent input size across all categories
    - Preserved local defect context (overlap prevents edge defects being cut)

For segmentation: we only keep patches where at least `min_defect_ratio`
  of pixels are defective (otherwise the patch is essentially normal and
  adds noise to the defect segmentation training set).

Output structure:
  data/processed/patches/
    bottle/
      train/
        good/
          bottle_train_good_0000_patch_0_0.png
          bottle_train_good_0000_patch_0_1.png
          ...
      test/
        broken_large/
          bottle_test_broken_large_0000_patch_0_0.png
          ...
    ground_truth/
      broken_large/
        bottle_test_broken_large_0000_patch_0_0.png  ← binary mask patch

Usage:
    python src/data/tile.py
    python src/data/tile.py --category bottle  # single category
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Tiling logic ──────────────────────────────────────────────────────────────

def extract_patches(
    image: np.ndarray,
    patch_size: int,
    overlap: int,
) -> list[tuple[np.ndarray, int, int]]:
    """
    Slide a window across the image and extract patches.

    Returns list of (patch, row_start, col_start) tuples.
    row_start/col_start are the top-left pixel coordinates — used to match
    the corresponding mask patch later.
    """
    stride = patch_size - overlap
    h, w = image.shape[:2]
    patches = []

    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch = image[r:r + patch_size, c:c + patch_size]
            patches.append((patch, r, c))

    # Handle right/bottom edge: add one final patch aligned to the edge
    # so we don't miss pixels at the boundary.
    if h % stride != 0:
        for c in range(0, w - patch_size + 1, stride):
            patch = image[h - patch_size:h, c:c + patch_size]
            patches.append((patch, h - patch_size, c))

    if w % stride != 0:
        for r in range(0, h - patch_size + 1, stride):
            patch = image[r:r + patch_size, w - patch_size:w]
            patches.append((patch, r, w - patch_size))

    return patches


def mask_has_enough_defect(
    mask: np.ndarray,
    patch_size: int,
    min_ratio: float,
) -> bool:
    """
    Returns True if the mask patch contains at least min_ratio
    defective pixels. Used to filter near-empty defect patches.

    mask values: 0 = normal, 255 = defective pixel.
    """
    total_pixels = patch_size * patch_size
    defect_pixels = np.sum(mask > 0)
    return (defect_pixels / total_pixels) >= min_ratio


# ── Per-category tiler ────────────────────────────────────────────────────────

def tile_category(
    category: str,
    raw_dir: Path,
    patches_dir: Path,
    patch_size: int,
    overlap: int,
    min_defect_ratio: float,
) -> dict:
    """
    Tiles all images for one MVTec category.
    Returns stats dict for logging.
    """
    cat_raw = raw_dir / category
    cat_patches = patches_dir / category

    stats = {
        "category": category,
        "train_patches": 0,
        "test_normal_patches": 0,
        "test_defect_patches": 0,
        "skipped_low_defect": 0,
    }

    # ── Train split (normal only) ──────────────────────────────────────────
    train_src = cat_raw / "train" / "good"
    train_dst = cat_patches / "train" / "good"
    train_dst.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(train_src.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        patches = extract_patches(img, patch_size, overlap)
        for patch, r, c in patches:
            stem = img_path.stem
            out_name = f"{category}_train_good_{stem}_patch_{r}_{c}.png"
            cv2.imwrite(str(train_dst / out_name), patch)
            stats["train_patches"] += 1

    # ── Test split ────────────────────────────────────────────────────────
    test_src = cat_raw / "test"
    gt_src = cat_raw / "ground_truth"

    for defect_dir in sorted(test_src.iterdir()):
        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name  # e.g. "good", "broken_large", "scratch"
        is_normal = defect_type == "good"

        img_dst = cat_patches / "test" / defect_type
        img_dst.mkdir(parents=True, exist_ok=True)

        # Ground truth masks exist only for defective images
        mask_src = gt_src / defect_type if not is_normal else None
        mask_dst = patches_dir / category / "ground_truth" / defect_type
        if not is_normal:
            mask_dst.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(defect_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Load the corresponding ground truth mask (if defective)
            mask = None
            if not is_normal and mask_src is not None:
                mask_path = mask_src / (img_path.stem + "_mask.png")
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            patches = extract_patches(img, patch_size, overlap)

            for patch, r, c in patches:
                stem = img_path.stem
                out_name = f"{category}_test_{defect_type}_{stem}_patch_{r}_{c}.png"

                # For defective patches: check defect coverage in mask
                if not is_normal and mask is not None:
                    mask_patches = extract_patches(mask, patch_size, overlap)
                    # Find the matching mask patch (same r, c)
                    matching_mask = next(
                        (mp for mp, mr, mc in mask_patches if mr == r and mc == c),
                        None,
                    )
                    if matching_mask is not None:
                        if not mask_has_enough_defect(matching_mask, patch_size, min_defect_ratio):
                            stats["skipped_low_defect"] += 1
                            continue
                        # Save mask patch alongside image patch
                        cv2.imwrite(str(mask_dst / out_name), matching_mask)

                cv2.imwrite(str(img_dst / out_name), patch)

                if is_normal:
                    stats["test_normal_patches"] += 1
                else:
                    stats["test_defect_patches"] += 1

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tile MVTec images into patches")
    parser.add_argument("--category", type=str, default=None,
                        help="Process a single category (default: all)")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    patches_dir = Path(cfg["paths"]["patches_dir"])
    patch_size = cfg["tiling"]["patch_size"]
    overlap = cfg["tiling"]["overlap"]
    min_defect_ratio = cfg["tiling"]["min_defect_ratio"]

    categories = (
        [args.category] if args.category
        else cfg["dataset"]["categories"]
    )

    print("=" * 60)
    print(f"  Tiling MVTec AD → {patch_size}×{patch_size} patches")
    print(f"  Overlap: {overlap}px | Min defect ratio: {min_defect_ratio}")
    print("=" * 60)

    all_stats = []
    for category in tqdm(categories, desc="Categories"):
        stats = tile_category(
            category=category,
            raw_dir=raw_dir,
            patches_dir=patches_dir,
            patch_size=patch_size,
            overlap=overlap,
            min_defect_ratio=min_defect_ratio,
        )
        all_stats.append(stats)
        tqdm.write(
            f"  {category:<16} train={stats['train_patches']:<6} "
            f"test_normal={stats['test_normal_patches']:<6} "
            f"test_defect={stats['test_defect_patches']:<6} "
            f"skipped={stats['skipped_low_defect']}"
        )

    # Save stats JSON for DVC metrics
    stats_path = Path(cfg["paths"]["metadata_dir"]) / "tiling_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    total_train = sum(s["train_patches"] for s in all_stats)
    total_defect = sum(s["test_defect_patches"] for s in all_stats)
    total_skipped = sum(s["skipped_low_defect"] for s in all_stats)

    print(f"\n  [OK] Tiling complete.")
    print(f"  Total train patches : {total_train}")
    print(f"  Total defect patches: {total_defect}")
    print(f"  Skipped (low defect): {total_skipped}")
    print(f"  Stats saved to      : {stats_path}")
    print("\n  Next step: python src/data/split.py")


if __name__ == "__main__":
    main()
