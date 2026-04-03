"""
src/training/losses.py

Custom loss functions for all three models.

  FocalLoss          → EfficientNet-B4 classifier
  DiceBCELoss        → U-Net segmentation
  (PatchCore has no loss — it uses nearest-neighbor scoring, no backprop)

Why not just use PyTorch's built-in losses?
  - CrossEntropyLoss treats all misclassifications equally.
    With 85% normal images, the model learns to always predict "normal".
    FocalLoss fixes this by focusing on hard, misclassified examples.

  - BCELoss alone for segmentation doesn't optimize mask overlap directly.
    DiceLoss does, but it's unstable on its own early in training.
    Combining them gets the best of both.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary and multi-class classification.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    The core idea in plain terms:
      Standard cross-entropy: loss = -log(p)  for all samples equally
      Focal loss:             loss = -(1-p)^gamma * log(p)

      The (1-p)^gamma term is the "focusing factor".
      - If the model is confident and correct (p → 1): (1-p)^gamma → 0
        → loss contribution is near zero → model ignores easy examples
      - If the model is wrong (p → 0): (1-p)^gamma → 1
        → loss contribution is full → model focuses on hard examples

    Parameters:
      alpha (float): class weight for the positive (defective) class.
                     0.25 means we slightly down-weight positives since
                     we're already using WeightedRandomSampler.
      gamma (float): focusing strength. 0 = standard cross-entropy.
                     2.0 is the value from the original paper.
      reduction:     "mean" (default) or "sum"
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw model outputs (before softmax)
            targets: [B]    integer class labels

        Returns:
            scalar loss
        """
        # Standard cross-entropy per sample (no reduction yet)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Get probability of the TRUE class for each sample
        # p_t = probability assigned to the correct label
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # Focusing factor: down-weights easy (high-confidence correct) examples
        focusing_factor = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        # alpha for positive class (defective=1), (1-alpha) for negative (normal=0)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        focal_loss = alpha_t * focusing_factor * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ── Dice Loss ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice coefficient = 2 * |A ∩ B| / (|A| + |B|)
      where A = predicted mask, B = ground truth mask

    Dice loss = 1 - Dice coefficient
    Perfect prediction → Dice = 1.0 → loss = 0.0
    No overlap at all  → Dice = 0.0 → loss = 1.0

    smooth: small constant to prevent division by zero on empty masks
            (when both pred and target are all zeros)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] raw model outputs (before sigmoid)
            targets: [B, H, W] or [B, 1, H, W] binary masks (0 or 1)

        Returns:
            scalar loss
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: [B, 1, H, W] → [B, H*W]
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# ── Combined Dice + BCE Loss ──────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss for segmentation.

    Why combine them?
      - BCE: optimizes per-pixel accuracy. Good early in training when
             predictions are noisy. Stable gradients.
      - Dice: directly optimizes mask overlap (the metric we actually care about).
              Better for class-imbalanced segmentation (most pixels are normal).
              Can be unstable early in training on its own.

    Combined: BCE stabilizes early training, Dice refines mask quality later.

    bce_pos_weight: up-weights defective pixels in BCE loss.
      If 90% of pixels are normal and 10% defective:
      pos_weight=9.0 makes each defective pixel count 9× more than normal.
      This prevents the model from predicting all-zeros (all normal) trivially.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        bce_pos_weight: float = 5.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        # pos_weight tensor — will be moved to correct device in forward()
        self.register_buffer(
            "pos_weight", torch.tensor([bce_pos_weight])
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] raw model outputs
            targets: [B, H, W] binary masks

        Returns:
            scalar loss, plus individual components for logging
        """
        targets_4d = targets.unsqueeze(1).float()  # [B, H, W] → [B, 1, H, W]

        # BCE with logits (numerically stable, no manual sigmoid)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets_4d,
            pos_weight=self.pos_weight,
        )

        dice = self.dice(logits, targets)

        total = self.bce_weight * bce + self.dice_weight * dice

        # Return dict so Lightning can log components separately
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice": dice.detach(),
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 4

    # Focal loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(B, 2)
    targets = torch.randint(0, 2, (B,))
    loss = focal(logits, targets)
    print(f"[FocalLoss]   loss={loss.item():.4f}  shape={loss.shape}")

    # Dice loss
    dice = DiceLoss()
    logits_seg = torch.randn(B, 1, 64, 64)
    masks = torch.randint(0, 2, (B, 64, 64)).float()
    loss = dice(logits_seg, masks)
    print(f"[DiceLoss]    loss={loss.item():.4f}")

    # Combined
    combined = DiceBCELoss(dice_weight=0.5, bce_weight=0.5, bce_pos_weight=5.0)
    out = combined(logits_seg, masks)
    print(f"[DiceBCELoss] total={out['loss'].item():.4f}  "
          f"bce={out['bce'].item():.4f}  dice={out['dice'].item():.4f}")

    print("\n[OK] All loss functions verified.")
