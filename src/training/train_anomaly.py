"""
src/training/train_anomaly.py

Local training entry point — PatchCore anomaly detection.

Unlike the classifier and segmentation model, PatchCore:
  - Has NO gradient training
  - Trains one model per MVTec category (15 total)
  - "Training" takes minutes, not hours
  - Uses anomalib internally

Usage:
    # All 15 categories
    python src/training/train_anomaly.py

    # Single category (faster for testing)
    python src/training/train_anomaly.py --category bottle
"""

import argparse
import os
import sys
sys.path.insert(0, "src")

import mlflow
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/train_anomaly.yaml")
    parser.add_argument("--category", default=None,
                        help="Single MVTec category (default: all 15)")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = load_config("configs/data.yaml")

    # Merge data paths into anomaly cfg
    cfg["paths"] = data_cfg["paths"]

    # SageMaker path override
    sm_data = os.environ.get("SM_CHANNEL_TRAINING")
    if sm_data:
        cfg["paths"]["raw_dir"] = sm_data
        print(f"[SageMaker] Data dir: {sm_data}")

    from models.anomaly import PatchCoreTrainer

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "defect_anomaly"))

    categories = (
        [args.category] if args.category
        else data_cfg["dataset"]["categories"]
    )

    print(f"\n{'='*50}")
    print(f"  Training PatchCore — {len(categories)} categories")
    print(f"  Backbone          : {cfg['model']['backbone']}")
    print(f"  Coreset ratio     : {cfg['model']['coreset_sampling_ratio']}")
    print(f"  k-NN neighbors    : {cfg['model']['num_neighbors']}")
    print(f"{'='*50}\n")

    # Train on first category to initialize trainer
    trainer = PatchCoreTrainer(cfg, category=categories[0])
    summary = trainer.train_all_categories(categories)

    print(f"\n  Mean Image AUROC: {summary['mean_image_auroc']:.4f}")
    print(f"  Mean Pixel AUROC: {summary['mean_pixel_auroc']:.4f}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════


"""
src/training/train_segmentation.py — embedded below for single-file delivery.
Run as: python src/training/train_segmentation.py
"""

SEGMENTATION_SCRIPT = '''
import argparse
import os
import sys
sys.path.insert(0, "src")

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg):
    sm_data   = os.environ.get("SM_CHANNEL_TRAINING")
    sm_model  = os.environ.get("SM_MODEL_DIR")
    if sm_data:
        cfg.setdefault("paths", {})["metadata_csv"] = (
            f"{sm_data}/metadata/dataset_metadata.csv"
        )
    if sm_model:
        cfg["checkpointing"]["dirpath"] = f"{sm_model}/segmentation"
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_segmentation.yaml")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = load_config("configs/data.yaml")
    cfg      = resolve_paths(cfg)

    from data.transforms import get_segmentation_transforms
    from data.dataset import DefectSegmentationDataset
    from models.segmentation import DefectSegmenter

    transforms   = get_segmentation_transforms(cfg["data"]["image_size"])
    metadata_csv = data_cfg["paths"]["metadata_csv"]
    dl_cfg       = data_cfg["dataloader"]

    def make_loader(split):
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

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches  : {len(val_loader)}")

    model = DefectSegmenter(cfg)

    mlflow_cfg    = cfg.get("mlflow", {})
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_cfg.get("experiment_name", "defect_segmentation"),
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"),
        tags=mlflow_cfg.get("run_tags", {}),
    )

    ckpt_cfg      = cfg["checkpointing"]
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_cfg["dirpath"],
        filename="segmenter-{epoch:02d}-{val_dice:.4f}",
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        save_top_k=ckpt_cfg["save_top_k"],
        save_last=True,
    )

    es_cfg       = cfg["training"]["early_stopping"]
    early_stop   = EarlyStopping(
        monitor=es_cfg["monitor"],
        patience=es_cfg["patience"],
        mode=es_cfg["mode"],
    )
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")

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

    print(f"\\n  Training U-Net Segmenter")
    print(f"  Epochs: {t_cfg[\'max_epochs\']}  |  Batch: {t_cfg[\'batch_size\']}  |  Precision: {t_cfg[\'precision\']}\\n")

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    print(f"\\n  Best checkpoint : {checkpoint_cb.best_model_path}")
    print(f"  Best val Dice   : {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
'''

# Write the segmentation script to its own file
import os
script_path = os.path.join(os.path.dirname(__file__), "train_segmentation.py")
with open(script_path, "w") as f:
    f.write('"""\nsrc/training/train_segmentation.py\n\nLocal training entry point — U-Net defect segmentation.\n\nUsage:\n    python src/training/train_segmentation.py\n    python src/training/train_segmentation.py --fast-dev-run\n"""\n')
    # Extract the actual script content (removing surrounding quotes)
    content = SEGMENTATION_SCRIPT.strip().strip("'")
    f.write(content)
