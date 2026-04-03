"""
src/serving/onnx_inference_engine.py

Hybrid inference engine for NullDefect serving:
- Classifier and segmenter from ONNX
- Anomaly (PatchCore) from checkpoint (.ckpt)
"""

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import yaml

from serving.inference_engine import InferenceEngine, InferenceResult


@dataclass
class _OnnxSessions:
    classifier: Any
    segmenter: Any
    classifier_input_name: str
    segmenter_input_name: str


class OnnxInferenceEngine:
    def __init__(
        self,
        classifier_onnx: str,
        segmenter_onnx: str,
        classifier_ckpt: Optional[str] = None,
        anomaly_ckpt: Optional[str] = None,
        anomaly_root: str = "checkpoints/anomaly",
        default_anomaly_category: str = "bottle",
        clf_train_config: str = "configs/train_classifier.yaml",
        device: str = "cpu",
        image_size: int = 512,
        anomaly_image_size: int = 256,
        clf_threshold: float = 0.5,
        seg_threshold: float = 0.5,
        anomaly_threshold: float = 0.5,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with: pip install onnxruntime"
            ) from exc

        self.image_size = int(image_size)
        self.anomaly_image_size = int(anomaly_image_size)
        self.clf_threshold = float(clf_threshold)
        self.seg_threshold = float(seg_threshold)
        self.anomaly_threshold = float(anomaly_threshold)
        self.device = device

        self.anomaly_root = anomaly_root
        self.default_anomaly_category = default_anomaly_category
        self.active_anomaly_category = default_anomaly_category
        self.anomaly_model = None
        self.anomaly_models: dict[str, Any] = {}
        self.gradcam_classifier = None
        self._cam = None
        self._ClassifierOutputTarget = None

        providers = ["CPUExecutionProvider"]
        if (
            device.startswith("cuda")
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        clf_session = ort.InferenceSession(classifier_onnx, providers=providers)
        seg_session = ort.InferenceSession(segmenter_onnx, providers=providers)

        self.sessions = _OnnxSessions(
            classifier=clf_session,
            segmenter=seg_session,
            classifier_input_name=clf_session.get_inputs()[0].name,
            segmenter_input_name=seg_session.get_inputs()[0].name,
        )

        self.classifier_onnx = classifier_onnx
        self.segmenter_onnx = segmenter_onnx

        if classifier_ckpt and Path(classifier_ckpt).exists():
            from models.classifier import DefectClassifier

            with open(clf_train_config) as f:
                clf_cfg = yaml.safe_load(f)

            self.gradcam_classifier = (
                DefectClassifier.load_from_checkpoint(
                    classifier_ckpt,
                    cfg=clf_cfg,
                    map_location=device,
                )
                .to(device)
                .eval()
            )
            OnnxInferenceEngine._setup_gradcam(self)

        if anomaly_ckpt and Path(anomaly_ckpt).exists():
            model = InferenceEngine._load_patchcore_model(anomaly_ckpt, device)
            self.anomaly_model = model
            self.anomaly_models[default_anomaly_category] = model

        print(f"[OnnxInferenceEngine] Ready on {device}")
        print(f"  Classifier ONNX: {classifier_onnx}")
        print(f"  Segmenter ONNX : {segmenter_onnx}")
        print(
            f"  Grad-CAM       : {'enabled' if self._cam is not None else 'disabled'}"
        )
        print(
            f"  Anomaly CKPT   : {'loaded' if self.anomaly_model is not None else 'not loaded (fallback to classifier proxy)'}"
        )

    def _setup_gradcam(self) -> None:
        if self.gradcam_classifier is None:
            return

        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            blocks_any: Any = getattr(self.gradcam_classifier.backbone, "blocks", None)
            if blocks_any is None:
                return
            target_layer = list(blocks_any)[-1]
            self._cam = GradCAMPlusPlus(
                model=self.gradcam_classifier,
                target_layers=[target_layer],
            )
            self._ClassifierOutputTarget = ClassifierOutputTarget
        except ImportError:
            print("[OnnxInferenceEngine] grad-cam not installed — heatmaps disabled")

    @staticmethod
    def _to_b64(image: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", image)
        return base64.b64encode(buf).decode("utf-8")

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_resized = cv2.resize(image_bgr, (self.image_size, self.image_size))
        image_rgb = (
            cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        )

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_rgb - mean) / std

        x = np.transpose(image_norm, (2, 0, 1))[None, ...].astype(np.float32)
        return x, raw_resized

    def _preprocess_anomaly(self, raw_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(
            image_rgb,
            (self.anomaly_image_size, self.anomaly_image_size),
            interpolation=cv2.INTER_AREA,
        )
        x = np.transpose(image_resized, (2, 0, 1))[None, ...].astype(np.float32) / 255.0
        return x

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _extract_seg_logits(self, outputs: list[np.ndarray]) -> np.ndarray:
        if not outputs:
            raise RuntimeError("Segmenter ONNX returned no outputs")

        out = outputs[0]
        if out.ndim == 4:
            out = out[0, 0]
        elif out.ndim == 3:
            out = out[0]
        elif out.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected segmenter output shape: {out.shape}")

        return out.astype(np.float32)

    def _classify(self, x: np.ndarray) -> dict:
        logits = self.sessions.classifier.run(
            None,
            {self.sessions.classifier_input_name: x},
        )[0]
        probs = self._softmax(logits)[0]

        prob_normal = float(probs[0])
        prob_defective = float(probs[1])
        is_defective = prob_defective >= self.clf_threshold

        return {
            "is_defective": is_defective,
            "confidence": prob_defective if is_defective else prob_normal,
            "defect_type": "defective" if is_defective else "normal",
            "probs": {
                "normal": round(prob_normal, 4),
                "defective": round(prob_defective, 4),
            },
            "prob_defective": prob_defective,
        }

    def _heatmap(self, x: np.ndarray, raw_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._cam is None or self._ClassifierOutputTarget is None:
            return None

        x_tensor = torch.from_numpy(x).to(self.device)
        cam_fn: Any = self._cam
        grayscale = cam_fn(input_tensor=x_tensor)[0]

        h, w = raw_bgr.shape[:2]
        cam_resized = cv2.resize(grayscale, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return cv2.addWeighted(raw_bgr, 0.6, heatmap, 0.4, 0)

    def _segment(self, x: np.ndarray, raw_bgr: np.ndarray) -> dict:
        outputs = self.sessions.segmenter.run(
            None,
            {self.sessions.segmenter_input_name: x},
        )
        logits = self._extract_seg_logits(outputs)
        probs = self._sigmoid(logits)

        mask_binary = (probs >= self.seg_threshold).astype(np.uint8) * 255
        defect_ratio = float((mask_binary > 0).sum()) / (
            self.image_size * self.image_size
        )

        mask_colored = np.zeros_like(raw_bgr)
        mask_colored[mask_binary > 0] = [0, 0, 255]
        seg_overlay = cv2.addWeighted(raw_bgr, 0.7, mask_colored, 0.3, 0)

        return {
            "mask": mask_binary,
            "seg_overlay": seg_overlay,
            "defect_ratio": defect_ratio,
        }

    def _find_anomaly_ckpt_for_category(self, category: str) -> Optional[str]:
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
        if not category:
            category = self.default_anomaly_category

        if category in self.anomaly_models:
            self.anomaly_model = self.anomaly_models[category]
            self.active_anomaly_category = category
            return True

        ckpt = self._find_anomaly_ckpt_for_category(category)
        if not ckpt:
            return False

        try:
            model = InferenceEngine._load_patchcore_model(ckpt, self.device)
            self.anomaly_models[category] = model
            self.anomaly_model = model
            self.active_anomaly_category = category
            print(
                f"[OnnxInferenceEngine] PatchCore loaded for category '{category}': {ckpt}"
            )
            return True
        except Exception as e:
            print(
                f"[OnnxInferenceEngine] WARNING: failed loading PatchCore for '{category}': {e}"
            )
            return False

    def has_anomaly_available(self, category: Optional[str] = None) -> bool:
        requested = category or self.default_anomaly_category
        if requested in self.anomaly_models:
            return True
        return self._find_anomaly_ckpt_for_category(requested) is not None

    def _anomaly_score(
        self,
        raw_bgr: np.ndarray,
        clf_prob_defective: float,
        anomaly_category: Optional[str],
    ) -> dict:
        requested = anomaly_category or self.default_anomaly_category
        self.set_anomaly_category(requested)

        if self.anomaly_model is not None:
            x = self._preprocess_anomaly(raw_bgr)
            x_tensor = torch.from_numpy(x).to(self.device)

            with torch.no_grad():
                out = self.anomaly_model(x_tensor)

            score = float(out.pred_score.flatten()[0].item())
            return {
                "score": score,
                "is_anomaly": score >= self.anomaly_threshold,
                "model_name": f"patchcore_wideresnet50[{self.active_anomaly_category}]",
            }

        score = float(clf_prob_defective)
        return {
            "score": score,
            "is_anomaly": score >= self.anomaly_threshold,
            "model_name": "classifier_proxy_onnx",
        }

    @staticmethod
    def _combined_viz(
        raw_bgr: np.ndarray,
        heatmap: Optional[np.ndarray],
        seg_overlay: np.ndarray,
        confidence: float,
        defect_ratio: float,
    ) -> np.ndarray:
        panels = [
            raw_bgr.copy(),
            heatmap if heatmap is not None else raw_bgr.copy(),
            seg_overlay,
        ]
        label_texts = [
            "Original",
            f"Grad-CAM++ (conf={confidence:.2f})"
            if heatmap is not None
            else f"Classifier (conf={confidence:.2f})",
            f"Segmentation (area={defect_ratio * 100:.1f}%)",
        ]

        labeled = []
        for panel, text in zip(panels, label_texts):
            p = panel.copy()
            overlay = p.copy()
            cv2.rectangle(overlay, (0, 0), (p.shape[1], 36), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, p, 0.4, 0, p)
            cv2.putText(
                p, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            labeled.append(p)

        return np.hstack(labeled)

    def predict(
        self, image_bgr: np.ndarray, anomaly_category: Optional[str] = None
    ) -> InferenceResult:
        t0 = time.perf_counter()

        x, raw_resized = self._preprocess(image_bgr)
        clf_result = self._classify(x)
        seg_result = self._segment(x, raw_resized)
        heatmap = self._heatmap(x, raw_resized)
        ano_result = self._anomaly_score(
            raw_bgr=raw_resized,
            clf_prob_defective=clf_result["prob_defective"],
            anomaly_category=anomaly_category,
        )

        combined = self._combined_viz(
            raw_bgr=raw_resized,
            heatmap=heatmap,
            seg_overlay=seg_result["seg_overlay"],
            confidence=clf_result["confidence"],
            defect_ratio=seg_result["defect_ratio"],
        )

        inference_time_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            is_defective=clf_result["is_defective"],
            defect_type=clf_result["defect_type"],
            confidence=clf_result["confidence"],
            classification_probs=clf_result["probs"],
            anomaly_score=ano_result["score"],
            is_anomaly=ano_result["is_anomaly"],
            defect_mask_b64=self._to_b64(seg_result["mask"]),
            defect_area_ratio=seg_result["defect_ratio"],
            heatmap_b64=self._to_b64(heatmap) if heatmap is not None else None,
            segmentation_b64=self._to_b64(seg_result["seg_overlay"]),
            combined_b64=self._to_b64(combined),
            inference_time_ms=inference_time_ms,
            image_size=(image_bgr.shape[1], image_bgr.shape[0]),
            model_versions={
                "classifier": Path(self.classifier_onnx).name,
                "segmenter": Path(self.segmenter_onnx).name,
                "anomaly": ano_result["model_name"],
            },
        )

    def predict_from_path(
        self,
        image_path: str,
        anomaly_category: Optional[str] = None,
    ) -> InferenceResult:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.predict(img, anomaly_category=anomaly_category)

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        anomaly_category: Optional[str] = None,
    ) -> InferenceResult:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return self.predict(img, anomaly_category=anomaly_category)
