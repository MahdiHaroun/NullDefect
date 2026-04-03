"""
src/data/transforms.py

Albumentations augmentation pipelines for all three models.

Three separate transform configs:
  1. classifier_transforms  — for EfficientNet-B4 (classification)
  2. anomaly_transforms     — for PatchCore (minimal aug — preserve normal appearance)
  3. segmentation_transforms — for U-Net (must apply same transform to image AND mask)

Key design decisions:
  - ImageNet normalization for all (EfficientNet, WideResNet pretrained on ImageNet)
  - Elastic distortion for defect realism (surface deformation on metal/fabric)
  - NO color jitter on anomaly model — it needs clean normal samples to build
    its memory bank; aggressive augmentation would corrupt the reference distribution
  - Segmentation: ALL spatial transforms applied jointly to image + mask
    (Albumentations handles this automatically with the `mask` target)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── ImageNet normalization (shared across all models) ─────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── 1. Classifier transforms (EfficientNet-B4) ────────────────────────────────

def get_classifier_transforms(image_size: int = 512) -> dict:
    """
    Returns train and val/test transforms for the classification model.

    Train: aggressive augmentation to prevent overfitting on the
           relatively small MVTec dataset (~3,600 defect images total).
    Val/Test: only resize + normalize — no augmentation.
    """
    train = A.Compose([
        A.Resize(image_size, image_size),

        # ── Geometric ──────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        # Affine replaces deprecated ShiftScaleRotate (Albumentations v2+)
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.4,
        ),

        # ── Elastic distortion ─────────────────────────────────────────
        # Simulates surface deformation on metal, fabric, leather.
        # Critical for defect generalization.
        A.ElasticTransform(
            alpha=120,
            sigma=6,
            p=0.3,
        ),

        # ── Color / texture ────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3,
        ),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # ── Cutout / CoarseDropout ─────────────────────────────────────
        # Forces model to not rely on single region — improves robustness
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(48, 64),
            hole_width_range=(48, 64),
            fill=0,
            p=0.2,
        ),

        # ── Normalize + to tensor ──────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    val_test = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return {"train": train, "val": val_test, "test": val_test}


# ── 2. Anomaly detection transforms (PatchCore / PaDiM) ──────────────────────

def get_anomaly_transforms(image_size: int = 512) -> dict:
    """
    Minimal augmentation for the anomaly detection model.

    PatchCore builds a memory bank of normal image features.
    Heavy augmentation distorts what "normal" looks like and degrades
    the quality of the memory bank. We only allow:
      - Safe flips (preserve surface texture orientation)
      - Resize + normalize

    No color jitter, no elastic, no noise.
    """
    train = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    val_test = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return {"train": train, "val": val_test, "test": val_test}


# ── 3. Segmentation transforms (U-Net) ───────────────────────────────────────

def get_segmentation_transforms(image_size: int = 512) -> dict:
    """
    Joint image + mask transforms for the segmentation model.

    CRITICAL: Every spatial transform (flip, rotate, elastic) must be
    applied identically to both the image and its ground truth mask.
    Albumentations handles this automatically when you pass
    `additional_targets={"mask": "mask"}` or use the `mask` key
    in the transform call.

    Color transforms (brightness, noise) are applied to image only —
    the mask is binary and must not be color-augmented.

    Usage:
        transformed = transform(image=img_array, mask=mask_array)
        image = transformed["image"]   # tensor
        mask  = transformed["mask"]    # tensor
    """
    train = A.Compose(
        [
            A.Resize(image_size, image_size),

            # ── Geometric (applied to both image AND mask) ──────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ElasticTransform(
                alpha=120,
                sigma=6,
                p=0.3,
            ),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                p=0.4,
            ),

            # ── Color (image only — mask is untouched by these) ─────────
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            # ── Normalize + to tensor ───────────────────────────────────
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        # This tells Albumentations that "mask" is a segmentation mask
        # and should receive only spatial transforms, not color transforms.
        additional_targets={"mask": "mask"},
    )

    val_test = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )

    return {"train": train, "val": val_test, "test": val_test}


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    dummy_img  = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

    # Classifier
    t = get_classifier_transforms()
    out = t["train"](image=dummy_img)
    print(f"[Classifier] image tensor shape: {out['image'].shape}")  # [3, 512, 512]

    # Anomaly
    t = get_anomaly_transforms()
    out = t["train"](image=dummy_img)
    print(f"[Anomaly]    image tensor shape: {out['image'].shape}")

    # Segmentation
    t = get_segmentation_transforms()
    out = t["train"](image=dummy_img, mask=dummy_mask)
    print(f"[Segmenter]  image tensor shape: {out['image'].shape}")
    print(f"[Segmenter]  mask  tensor shape: {out['mask'].shape}")   # [512, 512]

    print("\n[OK] All transforms verified.")
