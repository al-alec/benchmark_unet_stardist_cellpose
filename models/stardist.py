# stardist.py
# StarDist (paper-aligned logic) + Numba-accelerated per-pixel ray distances
# Key fixes:
# - Semantic head is pixel-wise and INCLUDES background (class 0).
#   => n_classes = n_nucleus_classes + 1
# - TTA8 ray-channel remapping is GEOMETRIC (robust for any n_rays).
# - Decode follows paper logic: (threshold candidates) -> polygons -> NMS -> shape refinement vote
# - Instance class computed by aggregating class probs over refined instance mask, excluding background.
#
# Notes:
# - Local maxima candidate selection is optional (often better in practice, not strictly required by paper).
# - Distance targets use discrete marching (fast), aligned with "distance to boundary along rays" definition.

from __future__ import annotations

import math
from math import pi
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numba import njit, prange
from scipy.ndimage import maximum_filter
from skimage.draw import polygon


# ============================================================
# Numba: per-pixel ray distances (discrete marching)
# ============================================================

@njit(parallel=True, fastmath=True)
def _stardist_dists_per_pixel_numba(mask: np.ndarray, n_rays: int, max_dist: int) -> np.ndarray:
    """
    For each pixel inside an instance, cast n_rays and measure distance (in pixels)
    until leaving that SAME instance (mask id changes).

    mask: (H,W) int32, 0=bg, >0 instance id
    out : (R,H,W) float32, bg=0
    """
    H, W = mask.shape
    out = np.zeros((n_rays, H, W), dtype=np.float32)

    angles = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    sin_a = np.sin(angles).astype(np.float32)
    cos_a = np.cos(angles).astype(np.float32)

    for y0 in prange(H):
        for x0 in range(W):
            inst = mask[y0, x0]
            if inst == 0:
                continue

            for k in range(n_rays):
                dy = sin_a[k]
                dx = cos_a[k]
                last_inside = 0

                # discrete steps of 1 pixel (fast).
                for step in range(1, max_dist + 1):
                    y = int(round(y0 + step * dy))
                    x = int(round(x0 + step * dx))
                    if y < 0 or y >= H or x < 0 or x >= W:
                        break
                    if mask[y, x] != inst:
                        break
                    last_inside = step

                out[k, y0, x0] = last_inside

    return out


def compute_star_distances_per_pixel(
    mask_inst: np.ndarray,
    n_rays: int = 64,
    max_dist: Optional[int] = 128,
) -> np.ndarray:
    """
    Numba-accelerated per-pixel distances.
    max_dist bounds runtime. If None, uses image diagonal.
    """
    mask = mask_inst.astype(np.int32, copy=False)
    H, W = mask.shape
    if max_dist is None:
        max_dist = int(math.ceil(math.sqrt(H * H + W * W)))
    return _stardist_dists_per_pixel_numba(mask, int(n_rays), int(max_dist))


def build_stardist_targets(
    mask_inst: np.ndarray,               # (H,W) instance ids, 0 bg
    mask_cls: Optional[np.ndarray],      # (H,W) semantic classes: 0 bg, 1..C nucleus types
    n_rays: int = 64,
    max_dist: Optional[int] = 128,
    n_nucleus_classes: Optional[int] = None,  # if provided, clamp classes to 0..C
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      fg_t    : (1,H,W) float32, 0/1
      dist_t  : (R,H,W) float32, bg=0
      cls_t   : (H,W) int64, semantic classes including bg=0 (if mask_cls is not None)
    """
    fg = (mask_inst > 0).astype(np.float32)
    dist = compute_star_distances_per_pixel(mask_inst, n_rays=n_rays, max_dist=max_dist).astype(np.float32)

    fg_t = torch.from_numpy(fg).unsqueeze(0)  # (1,H,W)
    dist_t = torch.from_numpy(dist)           # (R,H,W)

    cls_t = None
    if mask_cls is not None:
        cls = mask_cls.astype(np.int64, copy=False)
        if n_nucleus_classes is not None:
            C = int(n_nucleus_classes)
            cls = np.clip(cls, 0, C)  # 0..C where 0 is background
        cls_t = torch.from_numpy(cls)

    return fg_t, dist_t, cls_t


# ============================================================
# Model: U-Net depth 4 + 3 heads
# ============================================================

def conv_block(inp_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class StarDist(nn.Module):
    """
    U-Net depth 4 + heads:
      prob_logits : (B,1,H,W)
      dist_pos    : (B,R,H,W) (positive via softplus)
      class_logits: (B,(C+1),H,W) with background as class 0
    """

    def __init__(
        self,
        input_ch: int = 3,
        n_rays: int = 64,
        n_nucleus_classes: int = 6,
        base_ch: int = 32,
        max_dist: float = 80.0,
    ) -> None:
        super().__init__()

        self.max_dist = float(max_dist)
        self.n_rays = int(n_rays)
        self.n_nucleus_classes = int(n_nucleus_classes)
        self.n_classes = self.n_nucleus_classes + 1  # + background

        # Encoder
        self.enc1 = conv_block(input_ch, base_ch);        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_ch, base_ch * 2);     self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8); self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, 2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        # Heads
        self.head_prob = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.head_dist = nn.Conv2d(base_ch, self.n_rays, kernel_size=1)

        # Init for sigmoid * max_dist: aim for ~15-20px at start
        nn.init.kaiming_normal_(self.head_dist.weight, nonlinearity="linear")
        # nn.init.constant_(self.head_dist.bias, 0.0)  # sigmoid(0) = 0.5 → 40px with max_dist=80
        nn.init.constant_(self.head_dist.bias, -1.5)

        self.head_cls  = nn.Conv2d(base_ch, self.n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.enc1(x); p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3); p4 = self.pool4(c4)

        b = self.bottleneck(p4)

        u4 = self.up4(b);  u4 = torch.cat([u4, c4], dim=1); d4 = self.dec4(u4)
        u3 = self.up3(d4); u3 = torch.cat([u3, c3], dim=1); d3 = self.dec3(u3)
        u2 = self.up2(d3); u2 = torch.cat([u2, c2], dim=1); d2 = self.dec2(u2)
        u1 = self.up1(d2); u1 = torch.cat([u1, c1], dim=1); d1 = self.dec1(u1)

        prob_logits = self.head_prob(d1)

        # Sigmoid * max_dist with bias=0 starts at ~40px (sigmoid(0)=0.5)
        dist_pos = torch.sigmoid(self.head_dist(d1)) * self.max_dist

        class_logits = self.head_cls(d1)
        return prob_logits, dist_pos, class_logits




try:
    # meilleur que maximum_filter pour gérer les plateaux
    from skimage.feature import peak_local_max
    _HAS_SKIMAGE_PEAK = True
except Exception:
    _HAS_SKIMAGE_PEAK = False


def is_prob_map_flat(prob_map: np.ndarray, std_thr: float = 1e-3, range_thr: float = 1e-3):
    """
    Détecte une prob_map quasi-constante.
    Retourne (is_flat, stats_dict)
    """
    p = prob_map.astype(np.float32, copy=False)
    p_std = float(p.std())
    p_rng = float(p.max() - p.min())
    is_flat = (p_std < std_thr) or (p_rng < range_thr)
    return is_flat, {"std": p_std, "range": p_rng, "min": float(p.min()), "max": float(p.max())}


def pick_candidates(
    prob_map: np.ndarray,
    prob_thr: float,
    use_local_maxima: bool,
    local_max_footprint: int,
    max_candidates: int,
    plateau_mode: str = "skimage",   # "skimage" (si dispo) | "jitter" | "plain"
    jitter_eps: float = 1e-6,
    jitter_seed: int = 0,
):
    """
    Retourne (ys, xs) triés par score décroissant, avec cap max_candidates.
    - use_local_maxima=False => tous les pixels >= prob_thr
    - use_local_maxima=True => pics locaux robustes (anti-plateau)
    """
    H, W = prob_map.shape
    p = prob_map.astype(np.float32, copy=False)

    if not use_local_maxima:
        ys, xs = np.where(p >= float(prob_thr))
        if ys.size == 0:
            return ys, xs
        scores = p[ys, xs]
        order = np.argsort(scores)[::-1]
        if order.size > max_candidates:
            order = order[:max_candidates]
        return ys[order], xs[order]

    # ---- local maxima robust ----
    if plateau_mode == "skimage" and _HAS_SKIMAGE_PEAK:
        # min_distance ~ footprint//2 (proche de ta logique)
        coords = peak_local_max(
            p,
            threshold_abs=float(prob_thr),
            min_distance=max(1, int(local_max_footprint // 2)),
            exclude_border=False,
        )
        if coords.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        ys, xs = coords[:, 0], coords[:, 1]
        scores = p[ys, xs]
        order = np.argsort(scores)[::-1]
        if order.size > max_candidates:
            order = order[:max_candidates]
        return ys[order], xs[order]

    # fallback: jitter epsilon pour casser les plateaux
    if plateau_mode in ("jitter", "skimage"):
        rng = np.random.default_rng(jitter_seed)
        p_j = p + float(jitter_eps) * rng.standard_normal(p.shape).astype(np.float32)
    else:
        p_j = p

    footprint = np.ones((local_max_footprint, local_max_footprint), dtype=bool)
    max_f = maximum_filter(p_j, footprint=footprint, mode="constant", cval=-np.inf)
    is_max = (p_j == max_f)

    ys, xs = np.where((p >= float(prob_thr)) & is_max)
    if ys.size == 0:
        return ys, xs

    scores = p[ys, xs]
    order = np.argsort(scores)[::-1]
    if order.size > max_candidates:
        order = order[:max_candidates]
    return ys[order], xs[order]




# ============================================================
# Losses: BCE + masked MAE + CE + Tversky
# ============================================================

def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    if m.any():
        return x[m].mean()
    return x.mean() * 0.0



def tversky_loss_multiclass(
    class_logits: torch.Tensor,   # (B,C,H,W)
    cls_t: torch.Tensor,          # (B,H,W) int64 in 0..C-1
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
    include_background: bool = True,
) -> torch.Tensor:
    B, C, H, W = class_logits.shape
    prob = torch.softmax(class_logits, dim=1)
    onehot = F.one_hot(cls_t.clamp(0, C - 1), num_classes=C).permute(0, 3, 1, 2).float()

    tp = (prob * onehot).sum(dim=(0, 2, 3))
    fp = (prob * (1 - onehot)).sum(dim=(0, 2, 3))
    fn = ((1 - prob) * onehot).sum(dim=(0, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    if include_background:
        return 1.0 - tversky.mean()
    return 1.0 - tversky[1:].mean()


def stardist_losses(
    prob_logits: torch.Tensor,       # (B,1,H,W)
    dist_pos: torch.Tensor,          # (B,R,H,W) distances in pixels (>=0)
    class_logits: torch.Tensor,      # (B,C,H,W) C includes bg at 0
    fg_t: torch.Tensor,              # (B,1,H,W) float 0/1
    dist_t: torch.Tensor,            # (B,R,H,W) float targets in pixels
    cls_t: Optional[torch.Tensor] = None,  # (B,H,W) int64 0..C-1
    w_prob: float = 1.0,
    w_dist: float = 1.0,
    w_ce: float = 0.3,
    w_tversky: float = 0.3,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    tversky_include_background: bool = True,
    center_weight_floor: float = 0.2,
    pos_weight: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict:
      total, prob, dist, ce, tversky
    """
    device = prob_logits.device

    # --------------------
    # 1) Foreground prob loss
    # --------------------
    # loss_prob = F.binary_cross_entropy_with_logits(prob_logits, fg_t)
    # with torch.no_grad():
    #     # ratio négatifs/positifs sur le batch
    #     pos = fg_t.sum()
    #     neg = fg_t.numel() - pos
    #     pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=50.0)


    loss_prob = F.binary_cross_entropy_with_logits(prob_logits, fg_t,
                                                   pos_weight=torch.tensor(float(pos_weight), device=device))

    # --------------------
    # 2) Distance regression (masked on FG)
    #    log1p space stabilizes large values, and center-weight fixes "tiny distances everywhere"
    # --------------------
    fg_mask = (fg_t > 0.5)                       # (B,1,H,W) bool
    # fg_expand = fg_mask.expand_as(dist_pos)      # (B,R,H,W) bool
    #
    # # log regression
    # pred = torch.log1p(torch.clamp(dist_pos, min=0.0))
    # tgt  = torch.log1p(torch.clamp(dist_t,   min=0.0))
    # per_pix = torch.abs(pred - tgt)              # (B,R,H,W)

    fg_mask = (fg_t > 0.5)  # (B,1,H,W)
    fg_expand = fg_mask.expand_as(dist_pos)  # (B,R,H,W)

    pred = torch.log1p(torch.clamp(dist_pos, min=0.0))
    tgt = torch.log1p(torch.clamp(dist_t, min=0.0))
    per_pix = torch.abs(pred - tgt)  # L1 in log space

    # center weighting: emphasize central pixels (more informative rays)
    mean_t = dist_t.mean(dim=1, keepdim=True)  # (B,1,H,W)
    denom = mean_t.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    w = (center_weight_floor + (mean_t / denom)).expand_as(dist_pos)

    loss_dist = _masked_mean(per_pix * w, fg_expand)

    # --------------------
    # 3) Optional semantic losses
    # --------------------
    loss_ce = torch.tensor(0.0, device=device)
    loss_tv = torch.tensor(0.0, device=device)

    if cls_t is not None:
        loss_ce = F.cross_entropy(class_logits, cls_t)
        loss_tv = tversky_loss_multiclass(
            class_logits=class_logits,
            cls_t=cls_t,
            alpha=tversky_alpha,
            beta=tversky_beta,
            include_background=tversky_include_background,
        )

    # --------------------
    # 4) Total
    # --------------------
    total = (w_prob * loss_prob) + (w_dist * loss_dist)
    if cls_t is not None:
        total = total + (w_ce * loss_ce) + (w_tversky * loss_tv)

    return {
        "total": total,
        "prob": loss_prob,
        "dist": loss_dist,
        "ce": loss_ce,
        "tversky": loss_tv,
    }


# ============================================================
# Decode: polygons + NMS + shape refinement vote
# ============================================================

@dataclass
class PolyCandidate:
    score: float
    cy: int
    cx: int
    rr: np.ndarray
    cc: np.ndarray


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.float32)
    union = np.logical_or(a, b).sum(dtype=np.float32)
    return float(inter / union) if union > 0 else 0.0


def _poly_from_rays(
    cy: int,
    cx: int,
    d: np.ndarray,          # (R,)
    angles: np.ndarray,     # (R,)
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ys = cy + d * np.sin(angles)
    xs = cx + d * np.cos(angles)
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    rr, cc = polygon(ys, xs, shape=(H, W))
    return rr, cc


def stardist_decode(
    prob_map: np.ndarray,                     # (H,W) float in [0,1]
    dist_map: np.ndarray,                     # (R,H,W) float distances
    class_prob: Optional[np.ndarray] = None,  # (C,H,W) softmax probs including bg=0
    prob_thr: float = 0.5,
    nms_iou_thr: float = 0.3,
    use_local_maxima: bool = False,           # "paper strict" default
    local_max_footprint: int = 5,
    max_candidates: int = 5000,
    min_area: int = 10,
    vote_thr: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    1) candidate pixels from prob_map (paper: prob>=thr; optional local maxima)
    2) build polygons from rays
    3) NMS: winner + suppressed => shape refinement by vote
    4) instance class: aggregate class_prob over refined mask, choose among nucleus classes (1..C-1)
    """
    assert prob_map.ndim == 2
    assert dist_map.ndim == 3
    R, H2, W2 = dist_map.shape
    H, W = prob_map.shape
    if (H2, W2) != (H, W):
        raise ValueError("dist_map and prob_map must share spatial size")
    if class_prob is not None and class_prob.shape[1:] != (H, W):
        raise ValueError("class_prob must have shape (C,H,W)")

    angles = np.linspace(0.0, 2.0 * pi, R, endpoint=False).astype(np.float32)

    # candidate pixels
    # if use_local_maxima:
    #     footprint = np.ones((local_max_footprint, local_max_footprint), dtype=bool)
    #     max_f = maximum_filter(prob_map, footprint=footprint, mode="constant", cval=0.0)
    #     is_max = (prob_map == max_f)
    #     ys, xs = np.where((prob_map >= prob_thr) & is_max)
    # else:
    #     ys, xs = np.where(prob_map >= prob_thr)

    is_flat, flat_stats = is_prob_map_flat(prob_map, std_thr=1e-3, range_thr=1e-3)
    if is_flat:
        # On refuse de décoder si la map est plate: sinon plateau-maxima => confetti
        empty = np.zeros((H, W), dtype=np.int32)
        return empty, None if class_prob is None else np.zeros((1,), dtype=np.int32)

    # ---------- candidate pixels (robuste) ----------
    ys, xs = pick_candidates(
        prob_map=prob_map,
        prob_thr=prob_thr,
        use_local_maxima=use_local_maxima,
        local_max_footprint=local_max_footprint,
        max_candidates=max_candidates,
        plateau_mode="skimage",  # "skimage" si dispo, sinon jitter fallback
        jitter_eps=1e-6,
        jitter_seed=0,
    )

    if ys.size == 0:
        empty = np.zeros((H, W), dtype=np.int32)
        return empty, None if class_prob is None else np.zeros((1,), dtype=np.int32)

    if len(ys) == 0:
        empty = np.zeros((H, W), dtype=np.int32)
        return empty, None if class_prob is None else np.zeros((1,), dtype=np.int32)

    scores = prob_map[ys, xs].astype(np.float32)
    order = np.argsort(scores)[::-1]
    if len(order) > max_candidates:
        order = order[:max_candidates]

    candidates: List[PolyCandidate] = []
    for idx in order:
        cy = int(ys[idx]); cx = int(xs[idx])
        d = dist_map[:, cy, cx].astype(np.float32)
        if d.max() <= 1e-3:
            continue

        rr, cc = _poly_from_rays(cy, cx, d, angles, H, W)
        if len(rr) < min_area:
            continue

        candidates.append(PolyCandidate(score=float(prob_map[cy, cx]), cy=cy, cx=cx, rr=rr, cc=cc))

    if not candidates:
        empty = np.zeros((H, W), dtype=np.int32)
        return empty, None if class_prob is None else np.zeros((1,), dtype=np.int32)

    cand_masks: List[np.ndarray] = []
    for c in candidates:
        m = np.zeros((H, W), dtype=bool)
        m[c.rr, c.cc] = True
        cand_masks.append(m)

    suppressed = np.zeros((len(candidates),), dtype=bool)

    inst_map = np.zeros((H, W), dtype=np.int32)
    inst_classes: List[int] = [0]  # idx 0 unused
    current_id = 1

    for i in range(len(candidates)):
        if suppressed[i]:
            continue

        winner_mask = cand_masks[i]
        group_idxs = [i]
        suppressed[i] = True

        for j in range(i + 1, len(candidates)):
            if suppressed[j]:
                continue
            if _mask_iou(winner_mask, cand_masks[j]) >= nms_iou_thr:
                suppressed[j] = True
                group_idxs.append(j)

        # shape refinement vote (winner + suppressed)
        vote = np.zeros((H, W), dtype=np.float32)
        for gi in group_idxs:
            vote += cand_masks[gi].astype(np.float32)
        vote /= float(len(group_idxs))
        refined = (vote >= float(vote_thr))
        refined = refined & (inst_map == 0)

        if refined.sum() < min_area:
            continue

        inst_map[refined] = current_id

        if class_prob is not None:
            yy, xx = np.where(refined)
            if len(yy) == 0:
                inst_classes.append(0)
            else:
                s = class_prob[:, yy, xx].sum(axis=1)  # (C,)
                if s.shape[0] <= 1:
                    inst_classes.append(0)
                else:
                    nucleus_class = 1 + int(np.argmax(s[1:]))  # exclude bg=0
                    inst_classes.append(nucleus_class)

        current_id += 1

    inst_cls_arr = None
    if class_prob is not None:
        inst_cls_arr = np.array(inst_classes, dtype=np.int32)

    return inst_map, inst_cls_arr


# ============================================================
# TTA8: rot90 x hflip with GEOMETRIC ray remap (robust)
# ============================================================

def _tta_transform_np(img: np.ndarray, k: int, flip: bool) -> np.ndarray:
    x = np.rot90(img, k=k)  # CCW
    if flip:
        x = np.flip(x, axis=1)  # horizontal flip (left-right)
    return x


def _tta_inverse_prob(prob: np.ndarray, k: int, flip: bool) -> np.ndarray:
    x = prob
    if flip:
        x = np.flip(x, axis=1)
    x = np.rot90(x, k=-k)
    return x


def _tta_inverse_cls(cls: np.ndarray, k: int, flip: bool) -> np.ndarray:
    # cls: (C,H,W)
    x = cls
    if flip:
        x = np.flip(x, axis=2)
    x = np.rot90(x, k=-k, axes=(1, 2))
    return x


def _ray_perm_geometric(R: int, k: int, flip: bool) -> np.ndarray:
    """
    Build perm such that after inverse spatial transform,
      dist_orig[r] = dist_tta[perm[r]]
    where perm maps "original ray index" -> "ray index in augmented prediction".

    Robust approach:
    - Define original ray directions v = (cosθ, sinθ).
    - Apply FORWARD augmentation transform to v: rotate CCW by k*90, then hflip (x -> -x) if flip.
    - Convert transformed vector back to angle, map to nearest ray index in augmented coord system.
    """
    k = k % 4
    angles = np.linspace(0.0, 2.0 * np.pi, R, endpoint=False).astype(np.float64)

    # Rotation matrices for multiples of 90 degrees (CCW)
    if k == 0:
        Rm = np.array([[1.0, 0.0], [0.0, 1.0]])
    elif k == 1:
        Rm = np.array([[0.0, -1.0], [1.0, 0.0]])
    elif k == 2:
        Rm = np.array([[-1.0, 0.0], [0.0, -1.0]])
    else:  # k == 3
        Rm = np.array([[0.0, 1.0], [-1.0, 0.0]])

    Fm = np.array([[-1.0, 0.0], [0.0, 1.0]]) if flip else np.array([[1.0, 0.0], [0.0, 1.0]])

    perm = np.empty((R,), dtype=np.int64)
    for r in range(R):
        th = angles[r]
        v = np.array([np.cos(th), np.sin(th)], dtype=np.float64)

        # forward transform on directions: v_aug = F * (R * v)
        v_rot = Rm @ v
        v_aug = Fm @ v_rot

        th_aug = math.atan2(float(v_aug[1]), float(v_aug[0]))
        if th_aug < 0:
            th_aug += 2.0 * math.pi

        # nearest index on [0..R-1]
        idx = int(math.floor(th_aug / (2.0 * math.pi) * R + 0.5)) % R
        perm[r] = idx

    return perm


def _tta_inverse_dist(dist: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """
    dist: (R,H,W) predicted on augmented image.

    Steps:
      1) inverse spatial transform (flip, rot)
      2) remap ray channels via geometric permutation
    """
    x = dist
    if flip:
        x = np.flip(x, axis=2)
    x = np.rot90(x, k=-k, axes=(1, 2))

    R = x.shape[0]
    perm = _ray_perm_geometric(R, k=k, flip=flip)
    return x[perm, :, :]


# ============================================================
# Inference helpers
# ============================================================

@torch.no_grad()
def predict_maps(
    model: nn.Module,
    x: torch.Tensor,                # (1,C,H,W)
    apply_softmax_class: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (prob_map, dist_map, class_prob) as numpy for B=1.
    class_prob includes background channel 0.
    """
    model.eval()
    prob_logits, dist_pos, class_logits = model(x)

    prob_map = torch.sigmoid(prob_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    dist_map = dist_pos[0].detach().cpu().numpy().astype(np.float32)

    if apply_softmax_class:
        cls_prob = torch.softmax(class_logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
    else:
        cls_prob = class_logits[0].detach().cpu().numpy().astype(np.float32)

    return prob_map, dist_map, cls_prob


@torch.no_grad()
def predict_instances(
    model: nn.Module,
    x: torch.Tensor,                    # (1,C,H,W)
    prob_thr: float = 0.5,
    nms_iou_thr: float = 0.3,
    use_local_maxima: bool = False,
    local_max_footprint: int = 5,
    max_candidates: int = 5000,
    min_area: int = 10,
    vote_thr: float = 0.5,
    use_tta8: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Paper logic inference:
      - optional TTA8: rotations x horizontal flip, aggregate maps, then decode
    """
    if not use_tta8:
        prob, dist, cls_prob = predict_maps(model, x, apply_softmax_class=True)
        return stardist_decode(
            prob_map=prob,
            dist_map=dist,
            class_prob=cls_prob,
            prob_thr=prob_thr,
            nms_iou_thr=nms_iou_thr,
            use_local_maxima=use_local_maxima,
            local_max_footprint=local_max_footprint,
            max_candidates=max_candidates,
            min_area=min_area,
            vote_thr=vote_thr,
        )

    # TTA8
    x_np = x[0].detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
    prob_list: List[np.ndarray] = []
    dist_list: List[np.ndarray] = []
    cls_list: List[np.ndarray] = []

    R = getattr(model, "n_rays", None)
    if R is None:
        raise ValueError("model must have attribute n_rays")

    device = x.device
    for k in (0, 1, 2, 3):
        for flip in (False, True):
            xt = _tta_transform_np(x_np, k=k, flip=flip)
            xt_t = torch.from_numpy(xt.transpose(2, 0, 1)).unsqueeze(0).to(device).type_as(x)

            p, d, c = predict_maps(model, xt_t, apply_softmax_class=True)

            p0 = _tta_inverse_prob(p, k=k, flip=flip)
            d0 = _tta_inverse_dist(d, k=k, flip=flip)
            c0 = _tta_inverse_cls(c, k=k, flip=flip)

            prob_list.append(p0)
            dist_list.append(d0)
            cls_list.append(c0)

    prob_avg = np.mean(prob_list, axis=0)
    dist_avg = np.mean(dist_list, axis=0)
    cls_avg = np.mean(cls_list, axis=0)

    return stardist_decode(
        prob_map=prob_avg,
        dist_map=dist_avg,
        class_prob=cls_avg,
        prob_thr=prob_thr,
        nms_iou_thr=nms_iou_thr,
        use_local_maxima=use_local_maxima,
        local_max_footprint=local_max_footprint,
        max_candidates=max_candidates,
        min_area=min_area,
        vote_thr=vote_thr,
    )
