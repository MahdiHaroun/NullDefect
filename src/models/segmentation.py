"""
src/models/segmentation.py

U-Net + EfficientNet-B4 encoder — pixel-level defect segmentation.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import BinaryAUROC, BinaryJaccardIndex

from training.losses import DiceBCELoss


# ── Manual Dice metric — no torchmetrics.segmentation dependency ──────────────

class DiceMetric:
    """
    Simple stateful Dice coefficient metric.
    Replaces torchmetrics.Dice to avoid version conflicts.
    Dice = 2*TP / (2*TP + FP + FN)
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds   = (preds > self.threshold).long().flatten()
        targets = targets.long().flatten()
        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()

    def compute(self) -> torch.Tensor:
        denom = 2 * self.tp + self.fp + self.fn + 1e-7
        return torch.tensor(2 * self.tp / denom)


class DefectSegmenter(pl.LightningModule):
    """
    U-Net segmentation model with EfficientNet-B4 encoder.

    Input:  [B, 3, 512, 512]
    Output: [B, 1, 512, 512] logits (before sigmoid)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)

        m_cfg = cfg["model"]
        t_cfg = cfg["training"]
        l_cfg = cfg["loss"]

        self.freeze_epochs         = m_cfg.get("freeze_epochs", 2)
        self.learning_rate         = t_cfg["learning_rate"]
        self.encoder_lr_multiplier = t_cfg.get("encoder_lr_multiplier", 0.1)
        self.weight_decay          = t_cfg["weight_decay"]

        # ── Model ─────────────────────────────────────────────────────────
        self.model = smp.Unet(
            encoder_name=m_cfg["encoder"],
            encoder_weights= None,
            in_channels=m_cfg["in_channels"],
            classes=m_cfg["classes"],
            activation=m_cfg.get("activation"),
        )

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = DiceBCELoss(
            dice_weight=l_cfg.get("dice_weight", 0.5),
            bce_weight=l_cfg.get("bce_weight", 0.5),
            bce_pos_weight=l_cfg.get("bce_pos_weight", 5.0),
        )

        # ── Metrics — version-safe ─────────────────────────────────────────
        self.val_dice      = DiceMetric(threshold=0.5)
        self.test_dice     = DiceMetric(threshold=0.5)

        # BinaryJaccardIndex = same as JaccardIndex(task="binary") but explicit
        # Works on torchmetrics >= 0.10
        try:
            self.val_iou  = BinaryJaccardIndex(threshold=0.5)
            self.test_iou = BinaryJaccardIndex(threshold=0.5)
        except Exception:
            # Older torchmetrics fallback
            from torchmetrics import JaccardIndex
            self.val_iou  = JaccardIndex(num_classes=2, threshold=0.5)
            self.test_iou = JaccardIndex(num_classes=2, threshold=0.5)

        self.val_pix_auroc = BinaryAUROC()

    # ── Freeze encoder ────────────────────────────────────────────────────────

    def on_train_epoch_start(self):
        if self.current_epoch < self.freeze_epochs:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            if self.current_epoch == 0:
                print(f"\n[Segmenter] Encoder frozen for epochs 0..{self.freeze_epochs - 1}")
        else:
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            if self.current_epoch == self.freeze_epochs:
                print(f"\n[Segmenter] Encoder unfrozen — full fine-tuning begins")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits        = self(images)
        loss_dict     = self.criterion(logits, masks)

        self.log("train_loss",      loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_bce",  loss_dict["bce"],  on_epoch=True)
        self.log("train_loss_dice", loss_dict["dice"], on_epoch=True)
        return loss_dict["loss"]

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits        = self(images)
        loss_dict     = self.criterion(logits, masks)

        probs = torch.sigmoid(logits).squeeze(1)    # [B, H, W]
        preds = (probs > 0.5).long()

        self.val_dice.update(preds, masks.long())
        self.val_iou.update(preds, masks.long())
        self.val_pix_auroc.update(probs.flatten(), masks.long().flatten())

        self.log("val_loss", loss_dict["loss"], on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_dice",      self.val_dice.compute(),      prog_bar=True)
        self.log("val_iou",       self.val_iou.compute(),       prog_bar=True)
        self.log("val_pix_auroc", self.val_pix_auroc.compute(), prog_bar=True)

        self.val_dice.reset()
        self.val_iou.reset()
        self.val_pix_auroc.reset()

    # ── Test step ─────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits        = self(images)
        probs         = torch.sigmoid(logits).squeeze(1)
        preds         = (probs > 0.5).long()

        self.test_dice.update(preds, masks.long())
        self.test_iou.update(preds, masks.long())

    def on_test_epoch_end(self):
        self.log("test_dice", self.test_dice.compute())
        self.log("test_iou",  self.test_iou.compute())
        self.test_dice.reset()
        self.test_iou.reset()

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = [
            p for p in self.parameters()
            if not any(p is ep for ep in encoder_params)
        ]

        optimizer = torch.optim.AdamW([
            {"params": decoder_params, "lr": self.learning_rate},
            {"params": encoder_params, "lr": self.learning_rate * self.encoder_lr_multiplier},
        ], weight_decay=self.weight_decay)

        sched_cfg = self.hparams["training"]["scheduler"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg["T_0"],
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/train_segmentation.yaml") as f:
        cfg = yaml.safe_load(f)

    model      = DefectSegmenter(cfg)
    dummy_img  = torch.randn(2, 3, 512, 512)
    dummy_mask = torch.randint(0, 2, (2, 512, 512)).float()

    logits    = model(dummy_img)
    print(f"[DefectSegmenter] output shape : {logits.shape}")

    loss_dict = model.criterion(logits, dummy_mask)
    print(f"[DefectSegmenter] loss={loss_dict['loss'].item():.4f}  "
          f"bce={loss_dict['bce'].item():.4f}  "
          f"dice={loss_dict['dice'].item():.4f}")

    print(f"[DefectSegmenter] parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n[OK] DefectSegmenter verified.")