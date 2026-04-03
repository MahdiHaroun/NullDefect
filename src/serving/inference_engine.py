"""
src/serving/inference_engine.py

NullDefect Inference Engine — loads all three models once at startup,
runs inference on a single image, returns a combined result.

Design:
  - Models are loaded ONCE into memory when the engine is initialized
  - Each call to .predict() runs all three models on the image
  - Thread-safe for FastAPI concurrent requests (models in eval mode, no grad)

Usage:
    engine = InferenceEngine.from_config("configs/evaluate.yaml")
    result = engine.predict(image_array)   # numpy BGR image
    print(result.to_dict())
"""

import base64
import enum
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, "src")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """
    Complete inference result for one image.
    All images are base64-encoded PNGs for easy JSON serialization.
    """
    # Classification
    is_defective:       bool
    defect_type:        str           # "normal" or defect name
    confidence:         float         # 0.0 - 1.0
    classification_probs: dict        # {"normal": 0.06, "defective": 0.94}

    # Anomaly detection
    anomaly_score:      float         # higher = more anomalous
    is_anomaly:         bool          # threshold applied

    # Segmentation
    defect_mask_b64:    Optional[str] # base64 PNG — binary mask
    defect_area_ratio:  float         # fraction of pixels flagged as defective

    # Visualizations
    heatmap_b64:        Optional[str] # base64 PNG — Grad-CAM++ overlay
    segmentation_b64:   Optional[str] # base64 PNG — mask overlay on image
    combined_b64:       Optional[str] # base64 PNG — all overlays combined

    # Meta
    inference_time_ms:  float
    image_size:         tuple
    model_versions:     dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "classification": {
                "is_defective":   self.is_defective,
                "defect_type":    self.defect_type,
                "confidence":     round(self.confidence, 4),
                "probs":          self.classification_probs,
            },
            "anomaly": {
                "score":      round(self.anomaly_score, 4),
                "is_anomaly": self.is_anomaly,
            },
            "segmentation": {
                "defect_area_ratio": round(self.defect_area_ratio, 4),
                "mask_b64":          self.defect_mask_b64,
            },
            "visualizations": {
                "heatmap_b64":      self.heatmap_b64,
                "segmentation_b64": self.segmentation_b64,
                "combined_b64":     self.combined_b64,
            },
            "meta": {
                "inference_time_ms": round(self.inference_time_ms, 2),
                "image_size":        self.image_size,
                "model_versions":    self.model_versions,
            },
        }


# ── Inference Engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Loads all three models once and runs combined inference on single images.

    Thread-safe: models are in eval() mode, no gradient computation.
    Models are loaded to the specified device at initialization.
    """

    def __init__(
        self,
        classifier:        nn.Module,
        segmenter:         nn.Module,
        anomaly_memory:    Optional[nn.Module],
        classifier_cfg:    dict,
        anomaly_root:      str = "checkpoints/anomaly",
        anomaly_category:  str = "bottle",
        device:            str = "cpu",
        anomaly_threshold: float = 0.5,
        clf_threshold:     float = 0.5,
        seg_threshold:     float = 0.5,
    ):
        self.classifier        = classifier.to(device).eval()
        self.segmenter         = segmenter.to(device).eval()
        self.anomaly_model     = anomaly_memory
        self.classifier_cfg    = classifier_cfg
        self.device            = device
        self.anomaly_threshold = anomaly_threshold
        self.clf_threshold     = clf_threshold
        self.seg_threshold     = seg_threshold
        self.anomaly_root      = anomaly_root
        self.default_anomaly_category = anomaly_category
        self.active_anomaly_category  = anomaly_category
        self.anomaly_models: dict[str, nn.Module] = {}
        if anomaly_memory is not None:
            self.anomaly_models[anomaly_category] = anomaly_memory

        # Preprocess transforms
        from data.transforms import get_classifier_transforms, get_segmentation_transforms
        self._clf_transform = get_classifier_transforms(512)["val"]
        self._seg_transform = get_segmentation_transforms(512)["val"]

        # Grad-CAM++ setup
        self._cam = None
        self._setup_gradcam()

        print(f"[InferenceEngine] Ready on {device}")
        print(f"  Classifier : {sum(p.numel() for p in classifier.parameters()):,} params")
        print(f"  Segmenter  : {sum(p.numel() for p in segmenter.parameters()):,} params")
        print(f"  Anomaly    : {'loaded' if anomaly_memory else 'not loaded'}")

    @staticmethod
    def _ensure_anomalib_checkpoint_compatibility() -> None:
        """
        Add compatibility shims for loading older anomalib Lightning checkpoints.
        """
        import anomalib

        # Some older checkpoints reference anomalib.PrecisionType.
        if not hasattr(anomalib, "PrecisionType"):
            class PrecisionType(str, enum.Enum):
                FLOAT32  = "float32"
                FLOAT16  = "float16"
                BFLOAT16 = "bfloat16"
                FP32     = "32-true"
                FP16     = "16-mixed"
                BF16     = "bf16-mixed"

            anomalib.PrecisionType = PrecisionType

        # PyTorch >= 2.6 defaults torch.load to weights_only=True.
        # anomalib lightning checkpoints require full object deserialization.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    @classmethod
    def _load_patchcore_model(cls, anomaly_ckpt: str, device: str) -> nn.Module:
        """Load one PatchCore checkpoint into eval mode on requested device."""
        cls._ensure_anomalib_checkpoint_compatibility()
        from anomalib.models import Patchcore

        return Patchcore.load_from_checkpoint(
            anomaly_ckpt,
            map_location=device,
        ).to(device).eval()

    def _find_anomaly_ckpt_for_category(self, category: str) -> Optional[str]:
        """Find anomaly checkpoint path for a category under anomaly_root."""
        base = Path(self.anomaly_root)
        if not base.exists():
            return None

        direct = (
            base
            / category
            / "Patchcore"
            / "MVTecAD"
            / category
            / "v0"
            / "weights"
            / "lightning"
            / "model.ckpt"
        )
        if direct.exists():
            return str(direct)

        candidates = sorted((base / category).glob("**/*.ckpt"))
        return str(candidates[0]) if candidates else None

    def set_anomaly_category(self, category: str) -> bool:
        """
        Activate anomaly model for a category.
        Lazy-loads and caches model on first use.
        """
        if not category:
            category = self.default_anomaly_category

        # Reuse cached model.
        if category in self.anomaly_models:
            self.anomaly_model = self.anomaly_models[category]
            self.active_anomaly_category = category
            return True

        ckpt = self._find_anomaly_ckpt_for_category(category)
        if not ckpt:
            return False

        try:
            model = self._load_patchcore_model(ckpt, self.device)
            self.anomaly_models[category] = model
            self.anomaly_model = model
            self.active_anomaly_category = category
            print(f"[InferenceEngine] PatchCore loaded for category '{category}': {ckpt}")
            return True
        except Exception as e:
            print(f"[InferenceEngine] WARNING: failed loading PatchCore for '{category}': {e}")
            return False

    def _setup_gradcam(self):
        """Initialize Grad-CAM++ on the classifier's last conv block."""
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            target_layer = self.classifier.backbone.blocks[-1]
            self._cam    = GradCAMPlusPlus(
                model=self.classifier,
                target_layers=[target_layer],
            )
            self._ClassifierOutputTarget = ClassifierOutputTarget
        except ImportError:
            print("[InferenceEngine] grad-cam not installed — heatmaps disabled")

    @classmethod
    def from_checkpoints(
        cls,
        classifier_ckpt:    str,
        segmenter_ckpt:     str,
        anomaly_ckpt:       Optional[str] = None,
        anomaly_root:       str = "checkpoints/anomaly",
        anomaly_category:   str = "bottle",
        clf_train_config:   str = "configs/train_classifier.yaml",
        seg_train_config:   str = "configs/train_segmentation.yaml",
        device:             str = "cpu",
        anomaly_threshold:  float = 0.5,
    ) -> "InferenceEngine":
        """
        Factory method — load all models from checkpoint paths.
        This is the main constructor you'll use in practice.
        """
        from models.classifier import DefectClassifier
        from models.segmentation import DefectSegmenter

        print(f"[InferenceEngine] Loading models on {device}...")

        with open(clf_train_config) as f:
            clf_cfg = yaml.safe_load(f)
        with open(seg_train_config) as f:
            seg_cfg = yaml.safe_load(f)

        classifier = DefectClassifier.load_from_checkpoint(
            classifier_ckpt, cfg=clf_cfg, map_location=device
        )
        segmenter = DefectSegmenter.load_from_checkpoint(
            segmenter_ckpt, cfg=seg_cfg, map_location=device
        )

        # PatchCore anomaly model (anomalib)
        anomaly_model = None
        if anomaly_ckpt and Path(anomaly_ckpt).exists():
            try:
                anomaly_model = cls._load_patchcore_model(anomaly_ckpt, device)

                print(f"[InferenceEngine] PatchCore loaded: {anomaly_ckpt}")
            except Exception as e:
                print(f"[InferenceEngine] WARNING: could not load PatchCore checkpoint: {e}")
                anomaly_model = None

        return cls(
            classifier=classifier,
            segmenter=segmenter,
            anomaly_memory=anomaly_model,
            classifier_cfg=clf_cfg,
            anomaly_root=anomaly_root,
            anomaly_category=anomaly_category,
            device=device,
            anomaly_threshold=anomaly_threshold,
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, image_bgr: np.ndarray) -> dict:
        """
        Convert BGR numpy image to model-ready tensors.

        Returns dict with:
          clf_tensor: [1, 3, 512, 512] for classifier + Grad-CAM
          seg_tensor: [1, 3, 512, 512] for segmenter
          raw_resized: [512, 512, 3] BGR for overlays
        """
        image_rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_resized  = cv2.resize(image_bgr, (512, 512))

        clf_tensor = self._clf_transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)
        seg_tensor = self._seg_transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        return {
            "clf_tensor":  clf_tensor,
            "seg_tensor":  seg_tensor,
            "raw_resized": raw_resized,
        }

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, clf_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            logits = self.classifier(clf_tensor)
            probs  = F.softmax(logits, dim=1)[0]

        prob_defective = float(probs[1].item())
        prob_normal    = float(probs[0].item())
        is_defective   = prob_defective >= self.clf_threshold

        return {
            "is_defective":   is_defective,
            "confidence":     prob_defective if is_defective else prob_normal,
            "defect_type":    "defective" if is_defective else "normal",
            "probs": {
                "normal":    round(prob_normal, 4),
                "defective": round(prob_defective, 4),
            },
        }

    # ── Anomaly scoring ───────────────────────────────────────────────────────

    def _anomaly_score(
        self,
        raw_bgr: np.ndarray,
        clf_tensor: torch.Tensor,
        anomaly_category: Optional[str] = None,
    ) -> dict:
        """
        Compute anomaly score using PatchCore if loaded.
        Falls back to classifier defective probability proxy if unavailable.
        """
        requested_category = anomaly_category or self.default_anomaly_category
        self.set_anomaly_category(requested_category)

        if self.anomaly_model is not None:
            image_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            image_256 = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            image_tensor = (
                torch.from_numpy(image_256)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
                / 255.0
            )

            with torch.no_grad():
                out = self.anomaly_model(image_tensor)

            score = float(out.pred_score.flatten()[0].item())
            return {
                "score":      score,
                "is_anomaly": score >= self.anomaly_threshold,
            }

        # Fallback proxy score from classifier when PatchCore is unavailable.
        with torch.no_grad():
            logits = self.classifier(clf_tensor)
            probs  = F.softmax(logits, dim=1)[0]

        score = float(probs[1].item())

        return {
            "score":      score,
            "is_anomaly": score >= self.anomaly_threshold,
        }

    # ── Segmentation ──────────────────────────────────────────────────────────

    def _segment(self, seg_tensor: torch.Tensor, raw_bgr: np.ndarray) -> dict:
        with torch.no_grad():
            logits = self.segmenter(seg_tensor)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()  # [512, 512]

        mask_binary  = (probs >= self.seg_threshold).astype(np.uint8) * 255
        defect_ratio = float((mask_binary > 0).sum()) / (512 * 512)

        # Overlay: color defective pixels red on the original image
        mask_colored           = np.zeros_like(raw_bgr)
        mask_colored[mask_binary > 0] = [0, 0, 255]  # red = defect
        seg_overlay            = cv2.addWeighted(raw_bgr, 0.7, mask_colored, 0.3, 0)

        return {
            "mask":           mask_binary,
            "seg_overlay":    seg_overlay,
            "defect_ratio":   defect_ratio,
            "probs":          probs,
        }

    # ── Grad-CAM++ ────────────────────────────────────────────────────────────

    def _heatmap(self, clf_tensor: torch.Tensor, raw_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._cam is None:
            return None

        grayscale = self._cam(
            input_tensor=clf_tensor,
            targets=[self._ClassifierOutputTarget(1)],
        )[0]

        h, w        = raw_bgr.shape[:2]
        cam_resized = cv2.resize(grayscale, (w, h))
        heatmap     = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return cv2.addWeighted(raw_bgr, 0.6, heatmap, 0.4, 0)

    # ── Combined visualization ────────────────────────────────────────────────

    def _combined_viz(
        self,
        raw_bgr:    np.ndarray,
        heatmap:    Optional[np.ndarray],
        seg_overlay: np.ndarray,
        clf_result: dict,
        seg_result: dict,
    ) -> np.ndarray:
        """
        Three-panel composite: [Original | Grad-CAM | Segmentation]
        with text labels.
        """
        panels = [raw_bgr.copy()]

        if heatmap is not None:
            panels.append(heatmap)
        else:
            panels.append(raw_bgr.copy())

        panels.append(seg_overlay)

        label_texts = [
            f"Original",
            f"Grad-CAM++ (conf={clf_result['confidence']:.2f})",
            f"Segmentation (area={seg_result['defect_ratio']*100:.1f}%)",
        ]

        labeled = []
        for panel, text in zip(panels, label_texts):
            p = panel.copy()
            # Semi-transparent label background
            overlay = p.copy()
            cv2.rectangle(overlay, (0, 0), (p.shape[1], 36), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, p, 0.4, 0, p)
            cv2.putText(p, text, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            labeled.append(p)

        return np.hstack(labeled)

    # ── Encode to base64 ──────────────────────────────────────────────────────

    @staticmethod
    def _to_b64(image: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", image)
        return base64.b64encode(buf).decode("utf-8")

    # ── Main predict method ───────────────────────────────────────────────────

    def predict(self, image_bgr: np.ndarray, anomaly_category: Optional[str] = None) -> InferenceResult:
        """
        Run all three models on a single BGR image.

        Args:
            image_bgr: numpy array [H, W, 3] in BGR format (OpenCV default)

        Returns:
            InferenceResult with classification, anomaly, segmentation,
            heatmap, and combined visualization.
        """
        t0 = time.perf_counter()

        # Preprocess
        tensors = self._preprocess(image_bgr)

        # Run all three models
        clf_result = self._classify(tensors["clf_tensor"])
        ano_result = self._anomaly_score(
            tensors["raw_resized"],
            tensors["clf_tensor"],
            anomaly_category=anomaly_category,
        )
        seg_result = self._segment(tensors["seg_tensor"], tensors["raw_resized"])
        heatmap    = self._heatmap(tensors["clf_tensor"], tensors["raw_resized"])

        # Build visualizations
        combined = self._combined_viz(
            raw_bgr=tensors["raw_resized"],
            heatmap=heatmap,
            seg_overlay=seg_result["seg_overlay"],
            clf_result=clf_result,
            seg_result=seg_result,
        )

        inference_time_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            # Classification
            is_defective=clf_result["is_defective"],
            defect_type=clf_result["defect_type"],
            confidence=clf_result["confidence"],
            classification_probs=clf_result["probs"],

            # Anomaly
            anomaly_score=ano_result["score"],
            is_anomaly=ano_result["is_anomaly"],

            # Segmentation
            defect_mask_b64=self._to_b64(seg_result["mask"]),
            defect_area_ratio=seg_result["defect_ratio"],

            # Visualizations
            heatmap_b64=self._to_b64(heatmap) if heatmap is not None else None,
            segmentation_b64=self._to_b64(seg_result["seg_overlay"]),
            combined_b64=self._to_b64(combined),

            # Meta
            inference_time_ms=inference_time_ms,
            image_size=(image_bgr.shape[1], image_bgr.shape[0]),
            model_versions={
                "classifier": "efficientnet_b4",
                "segmenter":  "unet_efficientnet_b4",
                "anomaly":    f"patchcore_wideresnet50[{self.active_anomaly_category}]",
            },
        )

    def predict_from_path(
        self,
        image_path: str,
        anomaly_category: Optional[str] = None,
    ) -> InferenceResult:
        """Load image from disk and run inference."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.predict(img, anomaly_category=anomaly_category)

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        anomaly_category: Optional[str] = None,
    ) -> InferenceResult:
        """Decode image bytes and run inference. Used by FastAPI."""
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return self.predict(img, anomaly_category=anomaly_category)
