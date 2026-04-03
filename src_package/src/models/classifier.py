"""
src/models/classifier.py

EfficientNet-B4 classifier — PyTorch Lightning module.

What this file contains:
  DefectClassifier — the full Lightning module wrapping EfficientNet-B4.
  Handles: forward pass, training step, validation step, optimizer,
           LR scheduler, metric logging (AUROC, F1, accuracy).

Lightning module vs plain PyTorch:
  Plain PyTorch: you write the training loop manually (for epoch, for batch,
                 loss.backward(), optimizer.step(), etc.)
  Lightning:     you define what happens in one step. Lightning handles
                 the loop, device placement, mixed precision, checkpointing,
                 gradient clipping — all via the Trainer config.
"""

import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, F1Score, Accuracy
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from training.losses import FocalLoss


class DefectClassifier(pl.LightningModule):
    """
    EfficientNet-B4 binary defect classifier.

    Architecture:
      EfficientNet-B4 backbone (pretrained on ImageNet)
        → Global Average Pooling  (built into timm)
        → Dropout
        → Linear(1792, num_classes)   ← 1792 is B4's feature dim

    Training strategy:
      Epochs 0..freeze_epochs-1 : backbone frozen, only head trains
                                   (fast convergence of the new head)
      Epochs freeze_epochs..end  : full fine-tuning at lower LR
                                   (backbone adapts to X-ray/industrial domain)

    Metrics logged every step:
      train_loss, val_loss
      val_auroc  ← primary metric for checkpointing and early stopping
      val_f1
      val_acc
    """

    def __init__(self, cfg: dict):
        super().__init__()
        # save_hyperparameters stores cfg in self.hparams and logs it to MLflow
        self.save_hyperparameters(cfg)

        m_cfg = cfg["model"]
        t_cfg = cfg["training"]
        l_cfg = cfg["loss"]

        self.freeze_epochs = m_cfg.get("freeze_epochs", 2)
        self.learning_rate = t_cfg["learning_rate"]
        self.weight_decay  = t_cfg["weight_decay"]

        # ── Backbone ──────────────────────────────────────────────────────
        # timm.create_model downloads pretrained weights automatically.
        # num_classes=0 removes the default head — we add our own below.
        self.backbone = timm.create_model(
            m_cfg["name"],                    # "efficientnet_b4"
            pretrained=m_cfg["pretrained"],
            num_classes=0,                    # remove default classifier head
            drop_rate=m_cfg.get("drop_rate", 0.3),
            drop_path_rate=m_cfg.get("drop_path_rate", 0.2),
        )

        # Feature dimension output by the backbone
        # For EfficientNet-B4: 1792
        feature_dim = self.backbone.num_features

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=m_cfg.get("drop_rate", 0.3)),
            nn.Linear(feature_dim, m_cfg["num_classes"]),
        )

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            alpha=l_cfg.get("alpha", 0.25),
            gamma=l_cfg.get("gamma", 2.0),
        )

        # ── Metrics ───────────────────────────────────────────────────────
        # torchmetrics handles accumulation across batches automatically.
        num_classes = m_cfg["num_classes"]

        self.val_auroc = BinaryAUROC()
        self.val_f1    = BinaryF1Score()
        self.val_acc   = Accuracy(task="binary")

        self.test_auroc = BinaryAUROC()
        self.test_f1    = BinaryF1Score()

    # ── Freeze / unfreeze backbone ────────────────────────────────────────────

    def on_train_epoch_start(self):
        """
        Freeze backbone for the first `freeze_epochs` epochs.
        After that, unfreeze everything for full fine-tuning.

        This is called by Lightning at the start of each epoch automatically.
        """
        if self.current_epoch < self.freeze_epochs:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if self.current_epoch == 0:
                print(f"\n[Classifier] Backbone frozen for epochs 0..{self.freeze_epochs - 1}")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            if self.current_epoch == self.freeze_epochs:
                print(f"\n[Classifier] Backbone unfrozen — full fine-tuning begins")

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] normalized image tensor
        returns: [B, num_classes] logits (raw, before softmax)
        """
        features = self.backbone(x)   # [B, 1792]
        return self.head(features)    # [B, 2]

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        """
        Called by Lightning for each training batch.
        batch = (images, labels) from the DataLoader.
        Returns the loss scalar — Lightning calls .backward() on it.
        """
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        """
        Called for each validation batch. No gradients computed.
        We accumulate predictions across all batches, then compute
        metrics at epoch end (validation_epoch_end).
        """
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Probability of the positive (defective) class
        probs = torch.softmax(logits, dim=1)[:, 1]

        # Update metric accumulators (not computed yet — done at epoch end)
        self.val_auroc.update(probs, labels)
        self.val_f1.update(probs, labels)
        self.val_acc.update(probs, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """Compute and log accumulated metrics at the end of each val epoch."""
        auroc = self.val_auroc.compute()
        f1    = self.val_f1.compute()
        acc   = self.val_acc.compute()

        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_f1",    f1,    prog_bar=True)
        self.log("val_acc",   acc,   prog_bar=True)

        # Reset accumulators for next epoch
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_acc.reset()

    # ── Test step ─────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        self.test_auroc.update(probs, labels)
        self.test_f1.update(probs, labels)

    def on_test_epoch_end(self):
        self.log("test_auroc", self.test_auroc.compute())
        self.log("test_f1",    self.test_f1.compute())
        self.test_auroc.reset()
        self.test_f1.reset()

    # ── Optimizer & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        AdamW optimizer + CosineAnnealingWarmRestarts scheduler.

        AdamW = Adam + proper weight decay (decoupled from gradient update).
        Better generalization than Adam on fine-tuning tasks.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        sched_cfg = self.hparams["training"]["scheduler"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg["T_0"],
            T_mult=sched_cfg.get("T_mult", 1),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # step the scheduler once per epoch
                "monitor": "val_auroc",
            },
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/train_classifier.yaml") as f:
        cfg = yaml.safe_load(f)

    model = DefectClassifier(cfg)
    print(f"[DefectClassifier] backbone features: {model.backbone.num_features}")

    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    print(f"[DefectClassifier] output shape: {out.shape}")  # [2, 2]
    print(f"[DefectClassifier] parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n[OK] DefectClassifier verified.")
