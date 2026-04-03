"""
src/training/train_classifier.py

Local training entry point — EfficientNet-B4 classifier.

Runs on your local GPU (RTX 4050) or any CUDA device.
The same script is used by SageMaker — it just reads different
environment variables for data paths when running in the cloud.

Usage (local):
    python src/training/train_classifier.py

Usage (with custom config):
    python src/training/train_classifier.py --config configs/train_classifier.yaml

Usage (fast dev run — 2 batches per epoch, useful for testing):
    python src/training/train_classifier.py --fast-dev-run
"""

import argparse
import os
import socket
import json
from pathlib import Path
from urllib.parse import urlparse

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

# ── Path resolution: local vs SageMaker ──────────────────────────────────────
# SageMaker passes data/output paths as environment variables.
# Locally, we use the paths from config.
# This pattern lets the same script run in both environments.

def resolve_paths(cfg: dict) -> dict:
    """
    Override config paths with SageMaker environment variables if present.
    On local: SM_* vars are not set → falls back to config values.
    On SageMaker: SM_CHANNEL_TRAINING points to the S3 data channel.

    NOTE: This function handles output/checkpoint path overrides only.
    The metadata_csv path is resolved explicitly in main() via
    get_metadata_csv_path() to avoid silent fallback to local paths.
    """
    cfg = cfg.copy()
    sm_output = os.environ.get("SM_OUTPUT_DATA_DIR")
    sm_model = os.environ.get("SM_MODEL_DIR")

    if sm_output:
        cfg["checkpointing"]["dirpath"] = str(Path(sm_model or sm_output) / "classifier")
        print(f"[SageMaker] Output dir: {sm_output}")

    return cfg


def get_metadata_csv_path(data_cfg: dict) -> str:
    """
    Resolve the metadata CSV path with an explicit SM-first priority chain:

      1. SM_CHANNEL_TRAINING env var  →  <channel_root>/metadata/dataset_metadata.csv
      2. SM_TRAINING_ENV fallback     →  same construction from channel_input_dirs
      3. Local config                 →  data_cfg["paths"]["metadata_csv"]

    Using an explicit resolution here (instead of patching cfg["paths"] in
    resolve_paths) prevents the silent `or`-fallback to the local path that
    caused FileNotFoundError on SageMaker when train_classifier.yaml had no
    `paths:` section.
    """
    # Priority 1: standard SM channel env var
    sm_channel = os.environ.get("SM_CHANNEL_TRAINING")

    # Priority 2: SM_TRAINING_ENV JSON blob (some launchers skip SM_CHANNEL_*)
    if not sm_channel:
        try:
            sm_env_raw = os.environ.get("SM_TRAINING_ENV")
            if sm_env_raw:
                sm_channel = (
                    json.loads(sm_env_raw)
                    .get("channel_input_dirs", {})
                    .get("training")
                )
        except Exception:
            sm_channel = None

    if sm_channel:
        path = Path(sm_channel) / "metadata" / "dataset_metadata.csv"
        print(f"[SageMaker] Data channel  : {sm_channel}")
        print(f"[SageMaker] metadata_csv  : {path}")
        return str(path)

    # Priority 3: local config
    local_path = data_cfg["paths"]["metadata_csv"]
    print(f"[Local] metadata_csv: {local_path}")
    return local_path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _is_tracking_uri_reachable(uri: str, timeout_s: float = 2.0) -> bool:
    if not uri:
        return False
    if uri.startswith(("file:", "mlruns")):
        return True

    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def make_mlflow_logger(cfg: dict) -> MLFlowLogger:
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

    # On SageMaker, your laptop-hosted server won't be reachable.
    if os.environ.get("SM_TRAINING_ENV") and not _is_tracking_uri_reachable(tracking_uri):
        print(f"[WARN] MLflow tracking URI unreachable from SageMaker: {tracking_uri!r}")
        print("[WARN] Falling back to file-based MLflow tracking at 'mlruns'.")
        tracking_uri = "mlruns"

    return MLFlowLogger(
        experiment_name=mlflow_cfg.get("experiment_name", "defect_classifier"),
        tracking_uri=tracking_uri,
        tags=mlflow_cfg.get("run_tags", {}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_classifier.yaml")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run 2 batches per epoch for quick testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config("configs/data.yaml")
    cfg = resolve_paths(cfg)

    # ── Data ──────────────────────────────────────────────────────────────────
    # Import here (not top-level) so SageMaker container can override paths first
    import sys
    sys.path.insert(0, "src")

    from data.transforms import get_classifier_transforms
    from data.dataset import build_classification_loaders

    transforms = get_classifier_transforms(cfg["data"]["image_size"])

    # Explicit SM-aware path resolution — avoids silent fallback to local path
    metadata_csv = get_metadata_csv_path(data_cfg)

    loaders = build_classification_loaders(
        csv_path=metadata_csv,
        transforms=transforms,
        cfg=data_cfg,
        use_weighted_sampler=cfg["data"]["use_weighted_sampler"],
    )

    print(f"  Train batches : {len(loaders['train'])}")
    print(f"  Val batches   : {len(loaders['val'])}")
    print(f"  Test batches  : {len(loaders['test'])}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from models.classifier import DefectClassifier
    model = DefectClassifier(cfg)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_logger = make_mlflow_logger(cfg)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_cfg = cfg["checkpointing"]
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_cfg["dirpath"],
        filename="classifier-{epoch:02d}-{val_auroc:.4f}",
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        save_top_k=ckpt_cfg["save_top_k"],
        save_last=True,
    )

    es_cfg = cfg["training"]["early_stopping"]
    early_stop_cb = EarlyStopping(
        monitor=es_cfg["monitor"],
        patience=es_cfg["patience"],
        mode=es_cfg["mode"],
        min_delta=es_cfg.get("min_delta", 0.001),
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ───────────────────────────────────────────────────────────────
    t_cfg = cfg["training"]
    trainer = pl.Trainer(
        max_epochs=t_cfg["max_epochs"],
        precision=t_cfg["precision"],                # "16-mixed" = AMP
        gradient_clip_val=t_cfg["gradient_clip_val"],
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=mlflow_logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        # Detect unused parameters (helps catch bugs in Lightning modules)
        detect_anomaly=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Training EfficientNet-B4 Classifier")
    print(f"  Epochs     : {t_cfg['max_epochs']}")
    print(f"  Precision  : {t_cfg['precision']}")
    print(f"  Batch size : {t_cfg['batch_size']}")
    print(f"  Device     : {trainer.strategy.root_device if hasattr(trainer.strategy, 'root_device') else 'auto'}")
    print(f"{'='*50}\n")

    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\n  Running test set evaluation...")
    trainer.test(model, dataloaders=loaders["test"], ckpt_path="best")

    print(f"\n  Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"  Best val AUROC: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()