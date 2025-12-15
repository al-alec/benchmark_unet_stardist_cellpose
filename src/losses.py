from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(pred_logits, target, eps: float = 1e-6):
    """
    pred_logits : (B,1,H,W) logits
    target      : (B,1,H,W) float {0,1}
    """
    pred_prob = torch.sigmoid(pred_logits)
    pred_flat = pred_prob.view(pred_prob.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps
    dice = (2.0 * intersection + eps) / denom
    return 1.0 - dice.mean()


def bce_dice_loss(pred_logits, target):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    d = dice_loss(pred_logits, target)
    return bce + d

def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss for binary classification (logits).

    logits  : (B,1,H,W)
    targets : (B,1,H,W) float in {0,1}

    Returns scalar loss.
    """
    # BCE per-pixel
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = p if y=1 else (1-p)
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    # alpha factor
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction: {reduction}")


# ============================================================
# Distances regression (masked)
# ============================================================

def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    pred   : (B,R,H,W)
    target : (B,R,H,W)
    mask   : (B,1,H,W) or (B,H,W) in {0,1}
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=pred.dtype)

    # broadcast mask over rays
    mask_r = mask.expand_as(pred)

    l1 = (pred - target).abs() * mask_r
    denom = mask_r.sum().clamp_min(eps)
    return l1.sum() / denom


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Smooth L1 / Huber, masked.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=pred.dtype)
    mask_r = mask.expand_as(pred)

    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = loss * mask_r
    denom = mask_r.sum().clamp_min(eps)
    return loss.sum() / denom


# ============================================================
# StarDist loss (Route B)
# ============================================================

@dataclass
class StarDistLossConfig:
    w_prob: float = 1.0
    w_dist: float = 0.2

    # Prob head
    use_focal: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Dist head
    dist_loss: str = "smooth_l1"  # "l1" or "smooth_l1"
    smooth_l1_beta: float = 1.0

    # Stabilisation PanNuke
    clamp_dist: Optional[float] = None     # ex: 80.0 (pixels) or None
    normalize_dist: bool = False
    normalize_div: float = 64.0            # if normalize_dist=True, target/dist are divided by this


def stardist_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    w_prob: float = 1.0,
    w_dist: float = 0.2,
    *,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    dist_loss: str = "smooth_l1",
    smooth_l1_beta: float = 1.0,
    clamp_dist: Optional[float] = None,
    normalize_dist: bool = False,
    normalize_div: float = 64.0,
) -> torch.Tensor:
    """
    StarDist loss (Route B compatible).

    preds   : (B, 1+R, H, W)
              channel 0 = prob logits
              channels 1..R = positive distances (ideally already softplus'ed in model)
    targets : (B, 1+R, H, W)
              channel 0 = foreground/objectness target in {0,1}
              channels 1..R = radial distances per pixel (pixels)

    Key points:
      - Prob loss over ALL pixels
      - Distance regression ONLY over foreground pixels (mask = target_fg)
      - Optional clamp/normalization for stability
    """
    assert preds.ndim == 4 and targets.ndim == 4
    assert preds.shape == targets.shape, "preds and targets must have same shape"

    prob_logits = preds[:, :1]         # (B,1,H,W)
    dist_pred = preds[:, 1:]           # (B,R,H,W)

    fg = targets[:, :1].clamp(0, 1)    # (B,1,H,W)
    dist_tgt = targets[:, 1:]          # (B,R,H,W)

    # Optional distance clamp (both pred and target)
    if clamp_dist is not None:
        dist_pred = dist_pred.clamp(min=0.0, max=float(clamp_dist))
        dist_tgt = dist_tgt.clamp(min=0.0, max=float(clamp_dist))

    # Optional normalization (helps if distances are in large pixel units)
    if normalize_dist:
        div = float(normalize_div)
        dist_pred = dist_pred / div
        dist_tgt = dist_tgt / div

    # --- Prob loss ---
    if use_focal:
        prob_loss = focal_loss_with_logits(
            logits=prob_logits,
            targets=fg,
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean",
        )
    else:
        prob_loss = F.binary_cross_entropy_with_logits(prob_logits, fg)

    # --- Dist loss (masked on fg pixels) ---
    if dist_loss.lower() == "l1":
        dist_loss_val = masked_l1(dist_pred, dist_tgt, fg)
    elif dist_loss.lower() in ("smooth_l1", "huber"):
        dist_loss_val = masked_smooth_l1(dist_pred, dist_tgt, fg, beta=float(smooth_l1_beta))
    else:
        raise ValueError(f"Unknown dist_loss: {dist_loss}")

    total = w_prob * prob_loss + w_dist * dist_loss_val
    return total


# Optional convenience wrapper with config object
class StarDistLoss(nn.Module):
    def __init__(self, cfg: StarDistLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return stardist_loss(
            preds,
            targets,
            w_prob=c.w_prob,
            w_dist=c.w_dist,
            use_focal=c.use_focal,
            focal_alpha=c.focal_alpha,
            focal_gamma=c.focal_gamma,
            dist_loss=c.dist_loss,
            smooth_l1_beta=c.smooth_l1_beta,
            clamp_dist=c.clamp_dist,
            normalize_dist=c.normalize_dist,
            normalize_div=c.normalize_div,
        )
