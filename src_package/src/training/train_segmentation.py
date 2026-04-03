"""
src/training/train_segmentation.py
Local training entry point — U-Net defect segmentation.
"""

import argparse
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, "src")

import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_sm_channel() -> str | None:
    ch = os.environ.get("SM_CHANNEL_TRAINING")
    if ch:
        return ch
    try:
        raw = os.environ.get("SM_TRAINING_ENV")
        if raw:
            ch = json.loads(raw).get("channel_input_dirs", {}).get("training")
            if ch:
                return ch
    except Exception:
        pass
    return None


def get_metadata_csv_path(data_cfg: dict, sm_channel: str | None) -> str:
    if sm_channel:
        path = str(Path(sm_channel) / "metadata" / "dataset_metadata.csv")
        print(f"[SageMaker] metadata_csv : {path}")
        return path
    local = data_cfg["paths"]["metadata_csv"]
    print(f"[Local]     metadata_csv : {local}")
    return local


def remap_patch_paths(metadata_csv: str, sm_channel: str) -> str:
    """
    Rewrite local absolute paths in patch_path and mask_path columns
    to the SageMaker-mounted equivalent.

    Local:     /home/mahdi/.../data/processed/patches/bottle/...
    SageMaker: <SM_CHANNEL_TRAINING>/patches/bottle/...
    """
    df = pd.read_csv(metadata_csv)
    sm_patches = str(Path(sm_channel) / "patches")
    marker = "data/processed/patches"

    def remap(val):
        # Skip NaN, non-string, or already-remapped paths
        if not isinstance(val, str):
            return val
        idx = val.find(marker)
        if idx == -1:
            return val
        relative = val[idx + len(marker):]  # e.g. /bottle/test/broken_large/...
        return sm_patches + relative

    # Convert columns to object dtype first to avoid str accessor issues
    df["patch_path"] = df["patch_path"].astype(object).apply(remap)
    df["mask_path"]  = df["mask_path"].astype(object).apply(remap)

    remapped = tempfile.mktemp(suffix="_metadata_remapped.csv")
    df.to_csv(remapped, index=False)

    # Sanity check
    sample = df["patch_path"].dropna().iloc[0]
    print(f"[SageMaker] Paths remapped → {remapped}")
    print(f"[SageMaker] Sample patch   : {sample}")
    return remapped


def resolve_checkpoint_dir(cfg: dict) -> dict:
    cfg = cfg.copy()
    sm_model = os.environ.get("SM_MODEL_DIR")
    if sm_model:
        cfg["checkpointing"]["dirpath"] = str(Path(sm_model) / "segmentation")
        print(f"[SageMaker] checkpoint dir : {cfg['checkpointing']['dirpath']}")
    return cfg


def _is_tracking_uri_reachable(uri: str, timeout_s: float = 2.0) -> bool:
    if not uri or uri.startswith(("file:", "mlruns")):
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
    mlflow_cfg   = cfg.get("mlflow", {})
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    if os.environ.get("SM_TRAINING_ENV") and not _is_tracking_uri_reachable(tracking_uri):
        print(f"[WARN] MLflow URI unreachable — falling back to 'mlruns'")
        tracking_uri = "mlruns"
    return MLFlowLogger(
        experiment_name=mlflow_cfg.get("experiment_name", "defect_segmentation"),
        tracking_uri=tracking_uri,
        tags=mlflow_cfg.get("run_tags", {}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_segmentation.yaml")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = load_config("configs/data.yaml")
    cfg      = resolve_checkpoint_dir(cfg)

    # ── Resolve paths ──────────────────────────────────────────────────────
    sm_channel   = get_sm_channel()
    metadata_csv = get_metadata_csv_path(data_cfg, sm_channel)

    

    from data.transforms import get_segmentation_transforms
    from data.dataset import DefectSegmentationDataset
    from models.segmentation import DefectSegmenter

    transforms = get_segmentation_transforms(cfg["data"]["image_size"])
    dl_cfg     = data_cfg["dataloader"]

    def make_loader(split: str) -> DataLoader:
        ds = DefectSegmentationDataset(
            csv_path=metadata_csv,
            split=split,
            transform=transforms[split],
            defect_only=cfg["data"]["defect_only"],
        )
        return DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"],
            drop_last=(split == "train"),
        )

    train_loader = make_loader("train")
    val_loader   = make_loader("val")
    test_loader  = make_loader("test")

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    model         = DefectSegmenter(cfg)
    mlflow_logger = make_mlflow_logger(cfg)

    ckpt_cfg      = cfg["checkpointing"]
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_cfg["dirpath"],
        filename="segmenter-{epoch:02d}-{val_dice:.4f}",
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        save_top_k=ckpt_cfg["save_top_k"],
        save_last=True,
    )

    es_cfg     = cfg["training"]["early_stopping"]
    early_stop = EarlyStopping(
        monitor=es_cfg["monitor"],
        patience=es_cfg["patience"],
        mode=es_cfg["mode"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    t_cfg   = cfg["training"]
    trainer = pl.Trainer(
        max_epochs=t_cfg["max_epochs"],
        precision=t_cfg["precision"],
        gradient_clip_val=t_cfg["gradient_clip_val"],
        callbacks=[checkpoint_cb, early_stop, lr_monitor],
        logger=mlflow_logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )

    print(f"\n{'='*50}")
    print(f"  Training U-Net + EfficientNet-B4 Segmenter")
    print(f"  Epochs     : {t_cfg['max_epochs']}")
    print(f"  Precision  : {t_cfg['precision']}")
    print(f"  Batch size : {t_cfg['batch_size']}")
    print(f"{'='*50}\n")

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    print(f"\n  Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"  Best val Dice   : {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()