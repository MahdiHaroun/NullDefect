"""
steps/inference/inference_steps.py

ZenML steps for the inference pipeline.

Why wrap inference in ZenML steps?
  - Every prediction is a tracked artifact — you can replay any inference
  - Input images and outputs are versioned and linked
  - Latency and metadata logged to the dashboard automatically
  - Easy to swap models (just change the checkpoint path)
"""

import base64
import sys
from pathlib import Path
from typing import Annotated, Optional

import cv2
import numpy as np
from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

sys.path.insert(0, "src")

logger = get_logger(__name__)


# ── Step 1: Preprocess ────────────────────────────────────────────────────────

@step
def preprocess_step(
    image_path: str,
) -> Annotated[np.ndarray, "preprocessed_image"]:
    """
    Load image from disk and return as BGR numpy array.
    ZenML tracks the image path and array shape as artifacts.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    logger.info(f"Loaded image: {image_path} — shape={img.shape}")

    log_artifact_metadata(metadata={
        "image_path": image_path,
        "height":     img.shape[0],
        "width":      img.shape[1],
        "channels":   img.shape[2],
    })

    return img


# ── Step 2: Classify ──────────────────────────────────────────────────────────

@step(enable_cache=False)
def classify_step(
    image: np.ndarray,
    classifier_ckpt: str,
    train_config: str = "configs/train_classifier.yaml",
    threshold: float = 0.5,
    device: str = "cpu",
) -> Annotated[dict, "classification_result"]:
    """
    Run EfficientNet-B4 classifier on one image.
    Returns classification result dict.
    """
    import torch
    import torch.nn.functional as F
    import yaml
    from models.classifier import DefectClassifier
    from data.transforms import get_classifier_transforms

    device = device if torch.cuda.is_available() else "cpu"

    with open(train_config) as f:
        cfg = yaml.safe_load(f)

    model = DefectClassifier.load_from_checkpoint(
        classifier_ckpt, cfg=cfg, map_location=device
    ).to(device).eval()

    transforms  = get_classifier_transforms(512)
    image_rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor      = transforms["val"](image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    prob_defective = float(probs[1].item())
    prob_normal    = float(probs[0].item())
    is_defective   = prob_defective >= threshold

    result = {
        "is_defective": is_defective,
        "defect_type":  "defective" if is_defective else "normal",
        "confidence":   round(prob_defective if is_defective else prob_normal, 4),
        "probs": {
            "normal":    round(prob_normal, 4),
            "defective": round(prob_defective, 4),
        },
        "threshold": threshold,
    }

    logger.info(
        f"Classification: {'DEFECTIVE' if is_defective else 'NORMAL'} "
        f"(conf={result['confidence']:.4f})"
    )

    log_artifact_metadata(metadata={
        "is_defective": is_defective,
        "confidence":   result["confidence"],
        "defect_type":  result["defect_type"],
    })

    return result


# ── Step 3: Anomaly score ─────────────────────────────────────────────────────

@step(enable_cache=False)
def anomaly_step(
    image: np.ndarray,
    classifier_ckpt: str,
    train_config: str = "configs/train_classifier.yaml",
    threshold: float = 0.5,
    device: str = "cpu",
) -> Annotated[dict, "anomaly_result"]:
    """
    Compute anomaly score for one image.
    Uses classifier confidence as proxy anomaly score.
    For full PatchCore: load anomalib model per category.
    """
    import torch
    import torch.nn.functional as F
    import yaml
    from models.classifier import DefectClassifier
    from data.transforms import get_classifier_transforms

    device = device if torch.cuda.is_available() else "cpu"

    with open(train_config) as f:
        cfg = yaml.safe_load(f)

    model = DefectClassifier.load_from_checkpoint(
        classifier_ckpt, cfg=cfg, map_location=device
    ).to(device).eval()

    transforms = get_classifier_transforms(512)
    image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor     = transforms["val"](image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    score = float(probs[1].item())
    result = {
        "score":      round(score, 4),
        "is_anomaly": score >= threshold,
        "threshold":  threshold,
        "method":     "classifier_proxy",
    }

    logger.info(f"Anomaly score: {score:.4f} ({'ANOMALY' if result['is_anomaly'] else 'NORMAL'})")

    log_artifact_metadata(metadata={
        "anomaly_score": score,
        "is_anomaly":    result["is_anomaly"],
    })

    return result


# ── Step 4: Segment ───────────────────────────────────────────────────────────

@step(enable_cache=False)
def segment_step(
    image: np.ndarray,
    segmenter_ckpt: str,
    train_config: str = "configs/train_segmentation.yaml",
    threshold: float = 0.5,
    device: str = "cpu",
) -> Annotated[dict, "segmentation_result"]:
    """
    Run U-Net segmenter on one image.
    Returns binary mask, overlay, and defect area ratio.
    All images encoded as base64 PNG for JSON serialization.
    """
    import torch
    import yaml
    from models.segmentation import DefectSegmenter
    from data.transforms import get_segmentation_transforms

    device = device if torch.cuda.is_available() else "cpu"

    with open(train_config) as f:
        cfg = yaml.safe_load(f)

    model = DefectSegmenter.load_from_checkpoint(
        segmenter_ckpt, cfg=cfg, map_location=device
    ).to(device).eval()

    transforms = get_segmentation_transforms(512)
    image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raw_512    = cv2.resize(image, (512, 512))
    tensor     = transforms["val"](image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask_binary  = (probs >= threshold).astype(np.uint8) * 255
    defect_ratio = float((mask_binary > 0).sum()) / (512 * 512)

    # Overlay: red defect pixels on original
    mask_colored                      = np.zeros_like(raw_512)
    mask_colored[mask_binary > 0]     = [0, 0, 255]
    seg_overlay = cv2.addWeighted(raw_512, 0.7, mask_colored, 0.3, 0)

    def to_b64(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode("utf-8")

    result = {
        "defect_area_ratio": round(defect_ratio, 4),
        "mask_b64":          to_b64(mask_binary),
        "overlay_b64":       to_b64(seg_overlay),
        "threshold":         threshold,
    }

    logger.info(f"Segmentation: defect area = {defect_ratio*100:.2f}%")

    log_artifact_metadata(metadata={
        "defect_area_ratio": defect_ratio,
        "threshold":         threshold,
    })

    return result


# ── Step 5: Postprocess + combine ────────────────────────────────────────────

@step
def postprocess_step(
    image: np.ndarray,
    classification_result: dict,
    anomaly_result: dict,
    segmentation_result: dict,
    classifier_ckpt: str,
    train_config: str = "configs/train_classifier.yaml",
    device: str = "cpu",
) -> Annotated[dict, "inference_output"]:
    """
    Generate Grad-CAM++ heatmap and combine everything into final output dict.
    This is what gets returned to the FastAPI client.
    """
    import time
    import torch
    import yaml
    from models.classifier import DefectClassifier
    from data.transforms import get_classifier_transforms

    t0     = time.perf_counter()
    device = device if torch.cuda.is_available() else "cpu"

    def to_b64(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode("utf-8")

    raw_512    = cv2.resize(image, (512, 512))
    heatmap_b64 = None

    # Grad-CAM++
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import yaml

        with open(train_config) as f:
            cfg = yaml.safe_load(f)

        model = DefectClassifier.load_from_checkpoint(
            classifier_ckpt, cfg=cfg, map_location=device
        ).to(device).eval()

        transforms = get_classifier_transforms(512)
        image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor     = transforms["val"](image=image_rgb)["image"].unsqueeze(0).to(device)

        target_layer = model.backbone.blocks[-1]
        cam          = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        grayscale    = cam(tensor, targets=[ClassifierOutputTarget(1)])[0]

        cam_resized  = cv2.resize(grayscale, (512, 512))
        heatmap_bgr  = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_overlay = cv2.addWeighted(raw_512, 0.6, heatmap_bgr, 0.4, 0)
        heatmap_b64     = to_b64(heatmap_overlay)

    except Exception as e:
        logger.warning(f"Grad-CAM++ failed: {e}")

    # Combined three-panel visualization
    seg_overlay = cv2.imdecode(
        np.frombuffer(base64.b64decode(segmentation_result["overlay_b64"]), np.uint8),
        cv2.IMREAD_COLOR,
    )

    panels = [raw_512]
    labels = ["Original"]

    if heatmap_b64:
        heatmap_arr = cv2.imdecode(
            np.frombuffer(base64.b64decode(heatmap_b64), np.uint8), cv2.IMREAD_COLOR
        )
        panels.append(heatmap_arr)
        labels.append(f"Grad-CAM++ ({classification_result['confidence']:.2f})")

    panels.append(seg_overlay)
    labels.append(f"Segment ({segmentation_result['defect_area_ratio']*100:.1f}%)")

    labeled = []
    for panel, text in zip(panels, labels):
        p = panel.copy()
        overlay_bg = p.copy()
        cv2.rectangle(overlay_bg, (0, 0), (p.shape[1], 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.55, p, 0.45, 0, p)
        cv2.putText(p, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        labeled.append(p)

    combined_b64 = to_b64(np.hstack(labeled))

    inference_time_ms = (time.perf_counter() - t0) * 1000

    output = {
        "classification": classification_result,
        "anomaly":        anomaly_result,
        "segmentation":   segmentation_result,
        "visualizations": {
            "heatmap_b64":   heatmap_b64,
            "combined_b64":  combined_b64,
        },
        "meta": {
            "inference_time_ms": round(inference_time_ms, 2),
            "image_size":        (image.shape[1], image.shape[0]),
        },
    }

    logger.info(
        f"Inference complete — "
        f"{'DEFECTIVE' if classification_result['is_defective'] else 'NORMAL'} | "
        f"anomaly={anomaly_result['score']:.3f} | "
        f"defect_area={segmentation_result['defect_area_ratio']*100:.1f}% | "
        f"time={inference_time_ms:.1f}ms"
    )

    log_artifact_metadata(metadata={
        "is_defective":      classification_result["is_defective"],
        "confidence":        classification_result["confidence"],
        "anomaly_score":     anomaly_result["score"],
        "defect_area_ratio": segmentation_result["defect_area_ratio"],
        "inference_time_ms": round(inference_time_ms, 2),
    })

    return output
