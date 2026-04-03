"""
steps/evaluation/gradcam_step.py

ZenML step — generate Grad-CAM++ heatmaps for the classifier.

Output: list of saved PNG file paths, tracked as ZenML artifacts.
Each PNG is a composite: [original | heatmap overlay | ground truth mask]
"""

import sys
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from zenml import step
from zenml.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from steps.evaluation.torch_device import resolve_torch_device

logger = get_logger(__name__)


@step(enable_cache=False)
def gradcam_step(
    checkpoint_path: str,
    metadata_csv: str,
    output_dir: str = "reports/gradcam",
    train_config_path: str = "configs/train_classifier.yaml",
    samples_per_defect_type: int = 4,
    device: str = "cuda",
) -> Annotated[dict, "gradcam_outputs"]:
    """
    Generate Grad-CAM++ heatmaps for each defect type across all categories.

    Returns dict: {category/defect_type: [list of saved PNG paths]}
    ZenML tracks all paths as artifacts with full lineage.
    """
    from models.classifier import DefectClassifier
    from data.transforms import get_classifier_transforms
    from PIL import Image

    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        logger.warning("grad-cam not installed. Run: uv add grad-cam")
        logger.warning("Skipping Grad-CAM++ step.")
        return {"skipped": True, "reason": "grad-cam not installed"}

    import pandas as pd

    device = resolve_torch_device(device, log_warning=logger.warning)
    logger.info(f"Generating Grad-CAM++ heatmaps on {device}")

    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    model = DefectClassifier.load_from_checkpoint(
        checkpoint_path, cfg=train_cfg, map_location=device
    ).to(device).eval()

    # Target layer: last conv block of EfficientNet-B4
    target_layer = model.backbone.blocks[-1]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    transforms = get_classifier_transforms(512)
    df         = pd.read_csv(metadata_csv)
    test_df    = df[(df["split"] == "test") & (df["label"] == 1)]

    all_outputs = {}
    out_root    = Path(output_dir)

    for (category, defect_type), group in test_df.groupby(["category", "defect_type"]):
        if defect_type == "good":
            continue

        sample = group.sample(min(samples_per_defect_type, len(group)), random_state=42)
        saved  = []

        cat_dir = out_root / category / defect_type
        cat_dir.mkdir(parents=True, exist_ok=True)

        for _, row in sample.iterrows():
            raw_bgr = cv2.imread(row["patch_path"])
            if raw_bgr is None:
                continue

            raw_rgb    = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            pil_img    = Image.fromarray(raw_rgb)
            img_tensor = transforms["test"](image=raw_rgb)["image"].unsqueeze(0).to(device)

            # Confidence score
            with torch.no_grad():
                logits = model(img_tensor)
                prob   = float(F.softmax(logits, dim=1)[0, 1].item())

            # Grad-CAM++ heatmap
            grayscale_cam = cam(
                input_tensor=img_tensor,
                targets=[ClassifierOutputTarget(1)],
            )[0]

            h, w = raw_bgr.shape[:2]
            cam_resized = cv2.resize(grayscale_cam, (w, h))
            heatmap_bgr = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(raw_bgr, 0.6, heatmap_bgr, 0.4, 0)

            # Load ground truth mask if available
            panels = [raw_bgr, overlay]
            if pd.notna(row.get("mask_path")):
                mask_raw = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask_raw is not None:
                    mask_colored        = np.zeros_like(raw_bgr)
                    mask_colored[mask_raw > 127] = [0, 0, 255]
                    mask_panel          = cv2.addWeighted(raw_bgr, 0.6, mask_colored, 0.4, 0)
                    panels.append(mask_panel)

            # Add column labels
            label_texts = ["Original", "Grad-CAM++", "Ground Truth"]
            labeled = []
            for i, panel in enumerate(panels):
                p = panel.copy()
                cv2.putText(p, label_texts[i], (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                labeled.append(p)

            composite = np.hstack(labeled)

            stem     = Path(row["patch_path"]).stem
            out_name = f"{stem}_conf{prob:.2f}.png"
            out_path = cat_dir / out_name
            cv2.imwrite(str(out_path), composite)
            saved.append(str(out_path))

        if saved:
            key = f"{category}/{defect_type}"
            all_outputs[key] = saved
            logger.info(f"  [{key}] {len(saved)} heatmaps saved")

    total = sum(len(v) for v in all_outputs.values())
    logger.info(f"Total Grad-CAM++ images: {total}")
    logger.info(f"Saved to: {out_root.resolve()}")

    return all_outputs
