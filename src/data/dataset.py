"""
src/data/dataset.py

PyTorch Dataset classes for all three models.

Three datasets, one CSV:
  - DefectClassificationDataset  → EfficientNet-B4 (returns image + label)
  - AnomalyDataset               → PatchCore (returns image + is_anomaly flag)
  - DefectSegmentationDataset    → U-Net (returns image + pixel mask)

All three read from the same metadata CSV produced by split.py.
The difference is only what they return and how they handle the mask.

Also provides: DefectDataModule (PyTorch Lightning) — wraps all three
datasets into train/val/test DataLoaders with correct workers/pin_memory.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import yaml


# ── Config helper ─────────────────────────────────────────────────────────────

def _remap_paths(df: pd.DataFrame, sm_channel: str) -> pd.DataFrame:
    """
    Rewrite local-relative patch/mask paths to SageMaker channel root.
    e.g. data/processed/patches/... -> /opt/ml/input/data/training/patches/...
    """
    prefix = "data/processed/"
    channel_root = sm_channel.rstrip("/") + "/"
    df = df.copy()
    df["patch_path"] = df["patch_path"].astype(str).str.replace(prefix, channel_root, n=1, regex=False)
    if "mask_path" in df.columns:
        df["mask_path"] = df["mask_path"].fillna("").astype(str).str.replace(prefix, channel_root, n=1, regex=False)
        df["mask_path"] = df["mask_path"].replace("", float("nan"))
    return df

def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── 1. Classification Dataset ─────────────────────────────────────────────────

class DefectClassificationDataset(Dataset):
    """
    Returns (image_tensor, label) pairs for the EfficientNet-B4 classifier.

    Labels:
      0 = normal (good)
      1 = defective (any defect type — binary classification)

    For multi-class (defect type classification), use label_mode="multiclass"
    which maps each unique defect_type to an integer ID.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,                          # "train", "val", or "test"
        transform=None,
        label_mode: str = "binary",          # "binary" or "multiclass"
        category: Optional[str] = None,      # filter to one MVTec category
    ):
        self.transform = transform
        self.label_mode = label_mode

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if category:
            df = df[df["category"] == category].reset_index(drop=True)

        self.df = df
        sm_channel = os.environ.get("SM_CHANNEL_TRAINING")
        if sm_channel:
            self.df = _remap_paths(self.df, sm_channel)

        # Build defect type → integer ID mapping for multiclass mode
        if label_mode == "multiclass":
            all_types = sorted(df["defect_type"].unique())
            self.class_to_idx = {t: i for i, t in enumerate(all_types)}
            self.idx_to_class = {i: t for t, i in self.class_to_idx.items()}
            self.num_classes = len(all_types)
        else:
            self.num_classes = 2

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image (BGR → RGB)
        img = cv2.imread(row["patch_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Label
        if self.label_mode == "binary":
            label = torch.tensor(row["label"], dtype=torch.long)
        else:
            label = torch.tensor(
                self.class_to_idx[row["defect_type"]], dtype=torch.long
            )

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class weights for WeightedRandomSampler.
        Handles the extreme normal/defect imbalance in MVTec.

        Weight for each sample = 1 / count_of_its_class
        """
        if self.label_mode == "binary":
            labels = self.df["label"].values
        else:
            labels = self.df["defect_type"].map(self.class_to_idx).values

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return torch.from_numpy(sample_weights).float()


# ── 2. Anomaly Detection Dataset ──────────────────────────────────────────────

class AnomalyDataset(Dataset):
    """
    Dataset for PatchCore anomaly detection model.

    Training: returns ONLY normal (good) images.
      PatchCore trains exclusively on normal samples to build its memory bank.

    Validation/Test: returns all images with an is_anomaly flag.
      is_anomaly = 0 → normal, is_anomaly = 1 → defective

    Note: anomalib's PatchCore has its own data loading pipeline
    (MVTec datamodule). This dataset is for custom integration or
    when you want to use PatchCore outside of anomalib's CLI.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        transform=None,
        category: Optional[str] = None,
        normal_only: bool = False,   # force normal-only (used for train)
    ):
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if category:
            df = df[df["category"] == category].reset_index(drop=True)

        # During training: PatchCore only sees normal images
        if normal_only or split == "train":
            df = df[df["label"] == 0].reset_index(drop=True)

        self.df = df
        sm_channel = os.environ.get("SM_CHANNEL_TRAINING")
        if sm_channel:
            self.df = _remap_paths(self.df, sm_channel)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img = cv2.imread(row["patch_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": img,
            "is_anomaly": torch.tensor(row["label"], dtype=torch.float32),
            "category": row["category"],
            "defect_type": row["defect_type"],
        }


# ── 3. Segmentation Dataset ───────────────────────────────────────────────────

class DefectSegmentationDataset(Dataset):
    """
    Dataset for U-Net segmentation model.

    Returns (image_tensor, mask_tensor) pairs.
    mask_tensor is binary: 0 = normal pixel, 1 = defective pixel.

    Only defective patches are used (label == 1) because normal patches
    have no mask and contribute nothing to segmentation training.

    The joint transform (from transforms.py get_segmentation_transforms)
    is applied to both image and mask simultaneously.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        transform=None,
        category: Optional[str] = None,
        defect_only: bool = True,    # only use patches with masks
    ):
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if category:
            df = df[df["category"] == category].reset_index(drop=True)

        if defect_only:
            # Keep only patches that have a ground truth mask
            df = df[df["label"] == 1].dropna(subset=["mask_path"]).reset_index(drop=True)

        self.df = df
        sm_channel = os.environ.get("SM_CHANNEL_TRAINING")
        if sm_channel:
            self.df = _remap_paths(self.df, sm_channel)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image
        img = cv2.imread(row["patch_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load binary mask (255 = defect, 0 = normal → convert to 0/1)
        mask_path = row["mask_path"]
        if pd.isna(mask_path) or not Path(mask_path).exists():
            # Fallback: empty mask (all normal) — shouldn't happen with defect_only=True
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)  # binarize

        if self.transform:
            # Joint transform: both image and mask get the same spatial ops
            transformed = self.transform(image=img, mask=mask)
            img  = transformed["image"]   # [3, H, W] float tensor
            mask = transformed["mask"]    # [H, W] uint8 tensor
        else:
            img  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # Ensure mask is float for BCE loss
        mask = mask.float()

        return img, mask


# ── DataLoaders helper ────────────────────────────────────────────────────────

def build_classification_loaders(
    csv_path: str,
    transforms: dict,
    cfg: dict,
    category: Optional[str] = None,
    use_weighted_sampler: bool = True,
) -> dict:
    """
    Builds train/val/test DataLoaders for the classifier.

    WeightedRandomSampler on train set ensures balanced batches
    despite the heavy normal/defect imbalance.
    """
    loaders = {}
    batch_size = cfg["dataloader"]["batch_size"]
    num_workers = cfg["dataloader"]["num_workers"]
    pin_memory = cfg["dataloader"]["pin_memory"]

    for split in ["train", "val", "test"]:
        dataset = DefectClassificationDataset(
            csv_path=csv_path,
            split=split,
            transform=transforms[split],
            label_mode="binary",
            category=category,
        )

        sampler = None
        shuffle = split == "train"

        if split == "train" and use_weighted_sampler:
            weights = dataset.get_class_weights()
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False  # sampler and shuffle are mutually exclusive

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

    return loaders


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cfg = load_config()
    csv_path = cfg["paths"]["metadata_csv"]

    if not Path(csv_path).exists():
        print(f"[ERROR] Metadata CSV not found at {csv_path}")
        print("  Run split.py first.")
        sys.exit(1)

    # Test classification dataset
    ds = DefectClassificationDataset(csv_path=csv_path, split="train")
    print(f"[Classification] train samples: {len(ds)}")
    img, label = ds[0]
    print(f"  image shape: {img.shape}, label: {label.item()}")

    # Test anomaly dataset
    ds_a = AnomalyDataset(csv_path=csv_path, split="train")
    print(f"[Anomaly]        train samples (normal only): {len(ds_a)}")
    sample = ds_a[0]
    print(f"  image shape: {sample['image'].shape}, is_anomaly: {sample['is_anomaly'].item()}")

    # Test segmentation dataset
    ds_s = DefectSegmentationDataset(csv_path=csv_path, split="train")
    num_seg_samples = len(ds_s)
    print(f"[Segmentation]   train samples (defect+mask): {num_seg_samples}")

    if num_seg_samples > 0:
        img, mask = ds_s[0]
        print(f"  image shape: {img.shape}, mask shape: {mask.shape}")
        print(f"  mask unique values: {mask.unique()}")
    else:
        print("  [WARN] No segmentation samples found in the train split.")
        print("        This usually means there are no rows with:")
        print("          - split == 'train'")
        print("          - label == 1 (defective)")
        print("          - a valid non-empty mask_path")
        print("        Segmentation training will need such samples.")

    print("\n[OK] All datasets verified.")
