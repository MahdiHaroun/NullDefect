"""
src/serving/app.py

NullDefect FastAPI Inference Server

Exposes three endpoints:
  POST /predict       — send an image, get full inference result (JSON + base64 images)
  POST /predict/batch — send multiple images, get results for each
  POST /predict/onnx  — send an image, run ONNX classifier/segmenter + ckpt anomaly
  GET  /health        — health check + model status
  GET  /docs          — auto-generated Swagger UI (built into FastAPI)

The inference engine is loaded ONCE at startup and reused for all requests.
Models stay in memory — no checkpoint loading per request.

Usage:
    # Start server
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

    # Test with curl
    curl -X POST http://localhost:8000/predict \
         -F "file=@data/raw/mvtec/bottle/test/broken_large/000.png"

    # Test with Python
    import requests
    with open("image.png", "rb") as f:
        r = requests.post("http://localhost:8000/predict", files={"file": f})
    print(r.json())

    # View Swagger UI
    open http://localhost:8000/docs
"""

import os
import sys
import time
import uuid
import re
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, "src")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from serving.inference_engine import InferenceEngine
from serving.onnx_inference_engine import OnnxInferenceEngine


# ── Response schemas ──────────────────────────────────────────────────────────


class ClassificationResponse(BaseModel):
    is_defective: bool
    defect_type: str
    confidence: float
    probs: dict


class AnomalyResponse(BaseModel):
    score: float
    is_anomaly: bool
    threshold: float


class SegmentationResponse(BaseModel):
    defect_area_ratio: float
    mask_b64: Optional[str]
    overlay_b64: Optional[str]


class VisualizationResponse(BaseModel):
    heatmap_b64: Optional[str]
    combined_b64: Optional[str]


class MetaResponse(BaseModel):
    inference_time_ms: float
    image_size: tuple
    model_versions: dict


class PredictResponse(BaseModel):
    classification: ClassificationResponse
    anomaly: AnomalyResponse
    segmentation: SegmentationResponse
    visualizations: VisualizationResponse
    meta: MetaResponse


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    uptime_seconds: float
    device: str


# ── App setup ─────────────────────────────────────────────────────────────────

# Global engine — loaded once at startup
_engine: Optional[InferenceEngine] = None
_onnx_engine: Optional[OnnxInferenceEngine] = None
_start_time = time.time()


def _safe_stem(name: Optional[str]) -> str:
    if not name:
        return "image"
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "image"


def _save_combined_visualization(
    result_payload: dict,
    source_name: Optional[str],
    route_tag: str,
) -> Optional[str]:
    """Save combined_b64 image to disk and return the saved path."""
    combined_b64 = result_payload.get("visualizations", {}).get("combined_b64")
    if not combined_b64:
        return None

    output_dir = Path(os.environ.get("PREDICTIONS_SAVE_DIR", "outputs/predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = (
        f"{route_tag}_{_safe_stem(source_name)}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    )
    out_path = output_dir / filename

    try:
        out_path.write_bytes(base64.b64decode(combined_b64))
    except Exception as e:
        print(f"[NullDefect] WARNING: failed to save combined visualization: {e}")
        return None

    return str(out_path)


def _find_best_ckpt(directory: str) -> Optional[str]:
    """Auto-detect best checkpoint from directory."""
    d = Path(directory)
    if not d.exists():
        return None
    ckpts = [c for c in d.glob("*.ckpt") if "last" not in c.name]
    if not ckpts:
        ckpts = list(d.glob("*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.name, reverse=True)
    return str(ckpts[0])


def _find_anomaly_ckpt(anomaly_root: str, category: str) -> Optional[str]:
    """Find PatchCore Lightning checkpoint for a specific MVTec category."""
    base = Path(anomaly_root)
    if not base.exists():
        return None

    # Expected structure:
    # checkpoints/anomaly/<category>/Patchcore/MVTecAD/<category>/v0/weights/lightning/model.ckpt
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

    # Fallback: search first ckpt under that category subtree.
    candidates = sorted((base / category).glob("**/*.ckpt"))
    return str(candidates[0]) if candidates else None


def _find_onnx(root: str, patterns: list[str]) -> Optional[str]:
    """Find first ONNX file matching patterns under root."""
    base = Path(root)
    if not base.exists():
        return None

    for pattern in patterns:
        matches = sorted(base.glob(pattern), key=lambda p: p.name, reverse=True)
        if matches:
            return str(matches[0])
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models at startup, clean up at shutdown.
    FastAPI's lifespan context replaces the deprecated @app.on_event("startup").
    """
    global _engine, _onnx_engine

    print("\n[NullDefect] Loading inference models...")

    clf_ckpt = os.environ.get("CLASSIFIER_CKPT") or _find_best_ckpt(
        "checkpoints/classifier"
    )
    seg_ckpt = os.environ.get("SEGMENTER_CKPT") or _find_best_ckpt(
        "checkpoints/segmentation"
    )
    anomaly_root = os.environ.get("ANOMALY_ROOT", "checkpoints/anomaly")
    anomaly_category = os.environ.get("ANOMALY_CATEGORY", "bottle")
    anomaly_ckpt = os.environ.get("ANOMALY_CKPT") or _find_anomaly_ckpt(
        anomaly_root, anomaly_category
    )
    device = os.environ.get("INFERENCE_DEVICE", "cpu")

    clf_onnx = os.environ.get("ONNX_CLASSIFIER_PATH") or _find_onnx(
        "checkpoints",
        ["*classifier*.onnx", "classifier/*.onnx", "**/*classifier*.onnx"],
    )
    seg_onnx = os.environ.get("ONNX_SEGMENTER_PATH") or _find_onnx(
        "checkpoints",
        ["*segment*.onnx", "segmentation/*.onnx", "**/*segment*.onnx"],
    )
    if clf_ckpt and seg_ckpt:
        try:
            _engine = InferenceEngine.from_checkpoints(
                classifier_ckpt=clf_ckpt,
                segmenter_ckpt=seg_ckpt,
                anomaly_ckpt=anomaly_ckpt,
                anomaly_root=anomaly_root,
                anomaly_category=anomaly_category,
                device=device,
            )
            print(f"[NullDefect] Models loaded successfully on {device}")
            print(f"[NullDefect] PatchCore category: {anomaly_category}")
            print(
                f"[NullDefect] PatchCore ckpt: {anomaly_ckpt if anomaly_ckpt else 'not found'}"
            )
        except Exception as e:
            print(f"[NullDefect] WARNING: Could not load models: {e}")
            print(
                "[NullDefect] Server starting without models — /predict will return 503"
            )
            _engine = None
    else:
        print("[NullDefect] WARNING: Checkpoint paths not found")
        print(f"  Classifier: {clf_ckpt}")
        print(f"  Segmenter:  {seg_ckpt}")
        print("  Set CLASSIFIER_CKPT and SEGMENTER_CKPT env vars")
        _engine = None

    if clf_onnx and seg_onnx:
        try:
            _onnx_engine = OnnxInferenceEngine(
                classifier_onnx=clf_onnx,
                segmenter_onnx=seg_onnx,
                classifier_ckpt=clf_ckpt,
                anomaly_ckpt=anomaly_ckpt,
                anomaly_root=anomaly_root,
                default_anomaly_category=anomaly_category,
                device=device,
            )
            print(f"[NullDefect] ONNX models loaded successfully on {device}")
        except Exception as e:
            print(f"[NullDefect] WARNING: Could not load ONNX models: {e}")
            _onnx_engine = None
    else:
        print("[NullDefect] ONNX model paths not found")
        print(f"  ONNX classifier: {clf_onnx}")
        print(f"  ONNX segmenter : {seg_onnx}")
        print("  Set ONNX_CLASSIFIER_PATH and ONNX_SEGMENTER_PATH env vars")
        _onnx_engine = None

    yield  # server runs here

    # Shutdown
    print("[NullDefect] Shutting down inference server")
    _engine = None
    _onnx_engine = None


app = FastAPI(
    title="NullDefect — Industrial Defect Detection API",
    description="""
## NullDefect Inference API

Real-time defect detection using three complementary models:

- **EfficientNet-B4** — defect classification (normal vs defective)
- **PatchCore** — anomaly detection (unsupervised, catches unknown defects)
- **U-Net** — pixel-level defect segmentation (shows exactly where)

All responses include base64-encoded visualization images.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint.
    Returns model loading status and server uptime.
    """
    return HealthResponse(
        status="healthy" if _engine else "degraded",
        models_loaded={
            "classifier": _engine is not None,
            "segmenter": _engine is not None,
            "anomaly": _engine is not None and _engine.anomaly_model is not None,
            "onnx_classifier": _onnx_engine is not None,
            "onnx_segmenter": _onnx_engine is not None,
            "onnx_anomaly": _onnx_engine.has_anomaly_available()
            if _onnx_engine
            else False,
        },
        uptime_seconds=round(time.time() - _start_time, 1),
        device=_engine.device if _engine else "none",
    )


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(...),
    anomaly_category: Optional[str] = Query(
        None, description="MVTec category for PatchCore (e.g., bottle, cable, capsule)"
    ),
):
    """
    Run full defect detection on a single image.

    **Input:** multipart/form-data with `file` field (JPEG/PNG image)

    **Output:** JSON with:
    - `classification`: is_defective, defect_type, confidence
    - `anomaly`: anomaly score (0-1, higher = more abnormal)
    - `segmentation`: defect area ratio + binary mask (base64 PNG)
    - `visualizations`: Grad-CAM heatmap + combined 3-panel (base64 PNG)
    - `meta`: inference time, image size

    **Visualizations are base64-encoded PNGs.** Decode with:
    ```python
    import base64
    img_bytes = base64.b64decode(response["visualizations"]["combined_b64"])
    with open("result.png", "wb") as f: f.write(img_bytes)
    ```
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server logs for checkpoint paths.",
        )

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Run inference
    try:
        result = _engine.predict_from_bytes(
            image_bytes, anomaly_category=anomaly_category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    payload = result.to_dict()
    saved_path = _save_combined_visualization(payload, file.filename, "predict")
    if saved_path:
        payload["meta"]["saved_combined_path"] = saved_path

    return JSONResponse(content=payload)


@app.post("/predict/onnx", tags=["Inference"])
async def predict_onnx(
    file: UploadFile = File(...),
    anomaly_category: Optional[str] = Query(
        None,
        description="MVTec category for ONNX anomaly model (e.g., bottle, cable, capsule)",
    ),
):
    """
    Run defect detection using hybrid models.

    Uses:
    - ONNX classifier for class probabilities
    - ONNX segmenter for defect mask
    - PatchCore anomaly from checkpoint (category-based), with classifier proxy fallback
    """
    if _onnx_engine is None:
        raise HTTPException(
            status_code=503,
            detail="ONNX models not loaded. Set ONNX_CLASSIFIER_PATH and ONNX_SEGMENTER_PATH.",
        )

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = _onnx_engine.predict_from_bytes(
            image_bytes, anomaly_category=anomaly_category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference failed: {str(e)}")

    payload = result.to_dict()
    saved_path = _save_combined_visualization(payload, file.filename, "predict_onnx")
    if saved_path:
        payload["meta"]["saved_combined_path"] = saved_path

    return JSONResponse(content=payload)


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(
    files: list[UploadFile] = File(...),
    anomaly_category: Optional[str] = Query(
        None, description="MVTec category for PatchCore (applies to all files in batch)"
    ),
):
    """
    Run defect detection on multiple images.
    Returns a list of results in the same order as input files.
    Max 32 images per batch.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Max 32 images per batch")

    results = []
    for file in files:
        image_bytes = await file.read()
        try:
            result = _engine.predict_from_bytes(
                image_bytes, anomaly_category=anomaly_category
            )
            results.append({"filename": file.filename, "result": result.to_dict()})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(content={"results": results, "count": len(results)})


@app.post("/predict/path", tags=["Inference"])
async def predict_from_path(
    image_path: str,
    anomaly_category: Optional[str] = Query(
        None, description="MVTec category for PatchCore (e.g., bottle, cable, capsule)"
    ),
):
    """
    Run inference on an image file path (server-side path).
    Useful for testing with local MVTec images.

    Example: /data/raw/mvtec/bottle/test/broken_large/000.png
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    try:
        result = _engine.predict_from_path(
            image_path, anomaly_category=anomaly_category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result.to_dict())


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
