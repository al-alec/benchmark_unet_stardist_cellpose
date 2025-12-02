import torch
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
