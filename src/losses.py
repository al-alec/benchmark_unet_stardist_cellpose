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