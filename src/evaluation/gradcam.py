"""
src/evaluation/gradcam.py

Grad-CAM++ heatmap generator for the EfficientNet-B4 classifier.

What Grad-CAM++ does:
  1. Run a forward pass on an image → get class logits
  2. Backpropagate gradients to the TARGET LAYER (last conv block)
  3. Weight the activation maps by the gradient magnitudes
  4. Average weighted activations → coarse heatmap
  5. Upsample heatmap to original image size
  6. Overlay on the original image with a colormap

The result shows WHICH regions of the image most influenced the prediction.
For defect detection: the heatmap should light up on the actual defect.

Library: pytorch-grad-cam (pip install grad-cam)
  Implements Grad-CAM, Grad-CAM++, EigenCAM, etc.
  We use Grad-CAM++ — more accurate than vanilla Grad-CAM for
  multiple instances of the same class in one image.

Output: PNG images saved to reports/gradcam/
  Each image shows: [original | heatmap overlay | ground truth mask]
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml


def load_config(path: str = "configs/evaluate.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Target layer resolver ─────────────────────────────────────────────────────

def get_target_layer(model):
    """
    Resolve the target conv layer for Grad-CAM++ in EfficientNet-B4.

    We target the last block of the backbone — it has the best balance of:
      - High-level semantic information (knows what a defect is)
      - Enough spatial resolution to localize it

    timm's EfficientNet stores blocks in model.backbone.blocks
    The last block = model.backbone.blocks[-1]
    """
    # timm EfficientNet: backbone.blocks is a list of MBConv block groups
    return model.backbone.blocks[-1]


# ── Grad-CAM++ generator ──────────────────────────────────────────────────────

class GradCAMGenerator:
    """
    Generates Grad-CAM++ heatmaps for the defect classifier.

    Usage:
        gen = GradCAMGenerator(model, device="cuda")
        output_paths = gen.generate_batch(
            images=image_tensors,
            labels=label_tensors,
            raw_images=raw_bgr_images,
            mask_images=mask_arrays,
            output_dir="reports/gradcam/bottle/",
            prefix="bottle_broken",
        )
    """

    def __init__(self, model, device: str = "cuda", cfg: Optional[dict] = None):
        self.model  = model.to(device).eval()
        self.device = device
        self.cfg    = cfg or load_config()

        # Lazy import — only required if running evaluation
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            self._GradCAMPlusPlus      = GradCAMPlusPlus
            self._ClassifierOutputTarget = ClassifierOutputTarget
        except ImportError:
            raise ImportError(
                "grad-cam not installed. Run: pip install grad-cam"
            )

        target_layer = get_target_layer(model)
        self.cam = self._GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
        )

        self.alpha   = self.cfg["gradcam"]["alpha"]
        self.colormap_id = getattr(cv2, f"COLORMAP_{self.cfg['gradcam']['colormap'].upper()}")

    def generate_single(
        self,
        image_tensor: torch.Tensor,       # [3, H, W] normalized
        raw_image: np.ndarray,            # [H, W, 3] BGR uint8 (original, unnormalized)
        target_class: int = 1,            # 1 = defective class
        mask: Optional[np.ndarray] = None,# [H, W] binary mask (optional)
    ) -> np.ndarray:
        """
        Generate one Grad-CAM++ overlay image.

        Returns:
            [H, W*3, 3] composite: [original | heatmap overlay | mask (if provided)]
        """
        # ── Forward pass + CAM computation ────────────────────────────────
        input_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        targets = [self._ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]  # [H, W] float 0..1

        # ── Build heatmap overlay ──────────────────────────────────────────
        # Resize CAM to match raw image size (CAM is coarser than input)
        h, w = raw_image.shape[:2]
        grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))

        # Apply colormap (jet: blue=low, red=high activation)
        heatmap_bgr = cv2.applyColorMap(
            (grayscale_cam_resized * 255).astype(np.uint8),
            self.colormap_id,
        )

        # Blend heatmap with original image
        overlay = cv2.addWeighted(
            raw_image.astype(np.uint8), 1 - self.alpha,
            heatmap_bgr,                self.alpha,
            0,
        )

        # ── Build composite image ──────────────────────────────────────────
        panels = [raw_image.astype(np.uint8), overlay]

        if mask is not None:
            # Convert binary mask to BGR for visualization
            mask_vis = (mask * 255).astype(np.uint8)
            mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            # Color the defect pixels red
            mask_colored = np.zeros_like(raw_image)
            mask_colored[mask > 0] = [0, 0, 255]  # red = defect
            mask_panel = cv2.addWeighted(raw_image.astype(np.uint8), 0.6,
                                         mask_colored, 0.4, 0)
            panels.append(mask_panel)

        # Add column labels
        labels_text = ["Original", "Grad-CAM++", "Ground Truth"]
        labeled_panels = []
        for i, panel in enumerate(panels):
            labeled = panel.copy()
            cv2.putText(
                labeled, labels_text[i],
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )
            labeled_panels.append(labeled)

        composite = np.hstack(labeled_panels)
        return composite

    def generate_batch(
        self,
        images: list[torch.Tensor],
        raw_images: list[np.ndarray],
        masks: Optional[list[np.ndarray]],
        labels: list[int],
        output_dir: str,
        prefix: str = "sample",
    ) -> list[str]:
        """
        Generate Grad-CAM++ composites for a batch of images.
        Saves to output_dir, returns list of saved file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for i, (img_tensor, raw_img, label) in enumerate(zip(images, raw_images, labels)):
            mask = masks[i] if masks else None

            # Get model confidence for logging
            with torch.no_grad():
                logits = self.model(img_tensor.unsqueeze(0).to(self.device))
                prob   = F.softmax(logits, dim=1)[0, 1].item()

            composite = self.generate_single(
                image_tensor=img_tensor,
                raw_image=raw_img,
                target_class=1,
                mask=mask,
            )

            # Add confidence score to filename
            label_str = "defect" if label == 1 else "normal"
            filename  = f"{prefix}_{i:03d}_{label_str}_conf{prob:.2f}.png"
            out_path  = out_dir / filename

            cv2.imwrite(str(out_path), composite)
            saved_paths.append(str(out_path))

        print(f"  Grad-CAM++ saved {len(saved_paths)} images → {out_dir}")
        return saved_paths


# ── Dataset-level heatmap generation ─────────────────────────────────────────

def generate_gradcam_for_dataset(
    model,
    metadata_csv_path: str,
    raw_data_dir: str,
    output_dir: str,
    cfg: dict,
    device: str = "cuda",
) -> dict:
    """
    Generate Grad-CAM++ heatmaps for N samples per defect type
    across all MVTec categories.

    Returns dict mapping defect_type → list of saved image paths.
    """
    import pandas as pd
    import sys
    sys.path.insert(0, "src")
    from data.transforms import get_classifier_transforms

    transforms  = get_classifier_transforms(cfg["gradcam"]["image_size"])
    df          = pd.read_csv(metadata_csv_path)
    test_df     = df[(df["split"] == "test") & (df["label"] == 1)]  # defective only
    n_per_class = cfg["output"]["gradcam_samples_per_class"]

    generator   = GradCAMGenerator(model, device=device, cfg=cfg)
    all_outputs = {}

    # Group by category + defect_type
    for (category, defect_type), group in test_df.groupby(["category", "defect_type"]):
        if defect_type == "good":
            continue

        # Sample N images from this defect type
        sample = group.sample(min(n_per_class, len(group)), random_state=42)

        images, raw_images, masks_list = [], [], []

        for _, row in sample.iterrows():
            # Load raw BGR image (for overlay)
            raw_img = cv2.imread(row["patch_path"])
            if raw_img is None:
                continue

            # Apply transforms for model input
            rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            img_tensor = transforms["test"](image=rgb)["image"]

            # Load ground truth mask if available
            mask = None
            if pd.notna(row.get("mask_path")):
                mask_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
                if mask_raw is not None:
                    mask = (mask_raw > 127).astype(np.uint8)

            images.append(img_tensor)
            raw_images.append(raw_img)
            masks_list.append(mask)

        if not images:
            continue

        category_dir = Path(output_dir) / category / defect_type
        saved = generator.generate_batch(
            images=images,
            raw_images=raw_images,
            masks=masks_list,
            labels=[1] * len(images),
            output_dir=str(category_dir),
            prefix=f"{category}_{defect_type}",
        )
        all_outputs[f"{category}/{defect_type}"] = saved

    total = sum(len(v) for v in all_outputs.values())
    print(f"\n  Total Grad-CAM++ images generated: {total}")
    return all_outputs
