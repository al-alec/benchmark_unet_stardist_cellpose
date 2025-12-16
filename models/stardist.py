# stardist.py
from __future__ import annotations

import math
from math import pi
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import maximum_filter
from skimage.draw import polygon
from skimage.measure import regionprops


# ============================================================
# Utilities
# ============================================================

def compute_star_distances(
    mask_inst: np.ndarray,
    n_rays: int = 32,
) -> np.ndarray:
    """
    Route A: distances radiales EN PIXELS, constantes par instance.
    Pour chaque instance, on choisit un centre (pixel le plus proche du centroid),
    on tire n_rays rayons jusqu'à sortir de l'instance, puis on copie ce vecteur
    à tous les pixels de l'instance.

    Returns: (R,H,W) float32
    """
    assert mask_inst.ndim == 2
    H, W = mask_inst.shape
    dists = np.zeros((n_rays, H, W), dtype=np.float32)

    ids = np.unique(mask_inst)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return dists

    angles = np.linspace(0.0, 2.0 * pi, n_rays, endpoint=False).astype(np.float32)

    for prop in regionprops(mask_inst):
        inst_id = prop.label
        inst_mask = (mask_inst == inst_id)

        coords = prop.coords.astype(np.int32)  # (N,2)
        ys = coords[:, 0].astype(np.float32)
        xs = coords[:, 1].astype(np.float32)

        cy, cx = prop.centroid
        d2 = (ys - cy) ** 2 + (xs - cx) ** 2
        idx_center = int(np.argmin(d2))
        cy0 = int(ys[idx_center])
        cx0 = int(xs[idx_center])

        ray_dist = np.zeros(n_rays, dtype=np.float32)

        for k, theta in enumerate(angles):
            dy = float(math.sin(float(theta)))
            dx = float(math.cos(float(theta)))

            dist = 0.0
            step = 0
            while True:
                y = int(round(cy0 + step * dy))
                x = int(round(cx0 + step * dx))
                if y < 0 or y >= H or x < 0 or x >= W:
                    break
                if mask_inst[y, x] != inst_id:
                    break
                dist = float(step)
                step += 1

            ray_dist[k] = dist

        yy, xx = np.nonzero(inst_mask)
        dists[:, yy, xx] = ray_dist[:, None]

    return dists

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boolean masks."""
    inter = np.logical_and(a, b).sum(dtype=np.float32)
    union = np.logical_or(a, b).sum(dtype=np.float32)
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass
class PolyCandidate:
    score: float
    cy: int
    cx: int
    rr: np.ndarray
    cc: np.ndarray
    area: int


# ============================================================
# StarDist distances (Route B): per-pixel radial distances
# ============================================================

def compute_star_distances_per_pixel(
    mask_inst: np.ndarray,
    n_rays: int = 32,
    max_dist: Optional[int] = None,
) -> np.ndarray:
    """
    Compute StarDist radial distances PER PIXEL (Route B).

    Parameters
    ----------
    mask_inst : (H, W) int
        0 = background, >0 = instance id.
    n_rays : int
        Number of rays.
    max_dist : int or None
        Maximum distance (in pixels) to search along each ray.
        If None, uses image diagonal.

    Returns
    -------
    dists : (n_rays, H, W) float32
        For each pixel belonging to an instance, a vector of radial distances (pixels).
        Background stays 0.
    """
    assert mask_inst.ndim == 2
    H, W = mask_inst.shape
    dists = np.zeros((n_rays, H, W), dtype=np.float32)

    ids = np.unique(mask_inst)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return dists

    if max_dist is None:
        max_dist = int(math.ceil(math.sqrt(H * H + W * W)))

    angles = np.linspace(0.0, 2.0 * pi, n_rays, endpoint=False)
    sin_a = np.sin(angles).astype(np.float32)
    cos_a = np.cos(angles).astype(np.float32)

    # For each instance, compute distances for each pixel inside it
    for prop in regionprops(mask_inst):
        inst_id = prop.label
        coords = prop.coords.astype(np.int32)  # (N, 2) rows, cols

        # Build a fast lookup mask for this instance
        inst_mask = (mask_inst == inst_id)

        # For each pixel in the object, shoot rays
        for (y0, x0) in coords:
            # per-ray distance
            for k in range(n_rays):
                dy = float(sin_a[k])
                dx = float(cos_a[k])

                last_inside = 0
                # step from 1..max_dist
                for step in range(1, max_dist + 1):
                    y = int(round(y0 + step * dy))
                    x = int(round(x0 + step * dx))

                    if y < 0 or y >= H or x < 0 or x >= W:
                        break
                    if not inst_mask[y, x]:
                        break
                    last_inside = step

                dists[k, y0, x0] = float(last_inside)

    return dists


# ============================================================
# Basic U-Net blocks
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


# ============================================================
# StarDist model (U-Net + heads)
#   output:
#     - prob_logits: (B,1,H,W)
#     - dist_pos   : (B,R,H,W) positive distances
# ============================================================

class StarDist(nn.Module):
    """
    U-Net backbone + StarDist heads (Route B).
    - channel 0: objectness logits (foreground)
    - channels 1..R: raw distances -> enforced positive via softplus
    """

    def __init__(self, input_ch: int = 3, n_rays: int = 32, base_ch: int = 64) -> None:
        super().__init__()
        self.n_rays = n_rays

        # Encoder
        self.encoder1 = conv_block(input_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.decoder3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.decoder2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.decoder1 = conv_block(base_ch * 2, base_ch)

        # Heads
        self.out = nn.Conv2d(base_ch, 1 + n_rays, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.encoder1(x)
        p1 = self.pool1(c1)

        c2 = self.encoder2(p1)
        p2 = self.pool2(c2)

        c3 = self.encoder3(p2)
        p3 = self.pool3(c3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.decoder3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.decoder2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.decoder1(u1)

        raw = self.out(d1)  # (B, 1+R, H, W)

        prob_logits = raw[:, :1]
        dist_raw = raw[:, 1:]

        # force positive distances
        dist_pos = F.softplus(dist_raw)

        return torch.cat([prob_logits, dist_pos], dim=1)


# ============================================================
# Decode (Route B): candidates from prob_map + polygon NMS
# ============================================================

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
    prob_map: np.ndarray,
    dist_map: np.ndarray,
    prob_thr: float = 0.5,
    nms_iou_thr: float = 0.3,
    use_local_maxima: bool = True,
    local_max_footprint: int = 3,
    max_candidates: int = 5000,
    min_area: int = 10,
) -> np.ndarray:
    """
    StarDist decode (Route B): build polygon candidates for many pixels, then NMS by polygon IoU.

    Parameters
    ----------
    prob_map : (H, W) float32
        objectness probabilities (after sigmoid).
    dist_map : (R, H, W) float32
        positive radial distances per pixel.
    prob_thr : float
        probability threshold for candidate pixels.
    nms_iou_thr : float
        if IoU(candidate, kept) >= nms_iou_thr -> suppress candidate.
    use_local_maxima : bool
        if True, keep only local maxima of prob_map as candidates.
    local_max_footprint : int
        footprint for maximum_filter.
    max_candidates : int
        upper bound to avoid O(N^2) blowups.
    min_area : int
        min polygon area in pixels.

    Returns
    -------
    inst_map : (H, W) int32
        instance labels 0..K
    """
    assert prob_map.ndim == 2
    assert dist_map.ndim == 3

    R, H2, W2 = dist_map.shape
    H, W = prob_map.shape
    if (H2, W2) != (H, W):
        raise ValueError("dist_map and prob_map must share spatial size")

    angles = np.linspace(0.0, 2.0 * pi, R, endpoint=False).astype(np.float32)

    # Candidate pixels
    if use_local_maxima:
        footprint = np.ones((local_max_footprint, local_max_footprint), dtype=bool)
        max_f = maximum_filter(prob_map, footprint=footprint, mode="constant", cval=0.0)
        is_max = (prob_map == max_f)
        ys, xs = np.where((prob_map >= prob_thr) & is_max)
    else:
        ys, xs = np.where(prob_map >= prob_thr)

    if len(ys) == 0:
        return np.zeros((H, W), dtype=np.int32)

    scores = prob_map[ys, xs].astype(np.float32)
    order = np.argsort(scores)[::-1]

    # Cap candidates
    if len(order) > max_candidates:
        order = order[:max_candidates]

    # Build candidates (sorted by score)
    candidates: List[PolyCandidate] = []
    for idx in order:
        cy = int(ys[idx])
        cx = int(xs[idx])
        s = float(prob_map[cy, cx])

        d = dist_map[:, cy, cx].astype(np.float32)
        if d.max() <= 1e-3:
            continue

        rr, cc = _poly_from_rays(cy, cx, d, angles, H, W)
        area = int(len(rr))
        if area < min_area:
            continue

        candidates.append(PolyCandidate(score=s, cy=cy, cx=cx, rr=rr, cc=cc, area=area))

    if len(candidates) == 0:
        return np.zeros((H, W), dtype=np.int32)

    # NMS: keep best polygons by IoU on raster masks
    inst_map = np.zeros((H, W), dtype=np.int32)
    kept_masks: List[np.ndarray] = []

    current_id = 1
    for cand in candidates:
        # raster mask of candidate
        cand_mask = np.zeros((H, W), dtype=bool)
        cand_mask[cand.rr, cand.cc] = True

        suppressed = False
        for km in kept_masks:
            if _mask_iou(cand_mask, km) >= nms_iou_thr:
                suppressed = True
                break

        if suppressed:
            continue

        # accept
        inst_map[cand.rr, cand.cc] = current_id
        kept_masks.append(cand_mask)
        current_id += 1

    return inst_map


# ============================================================
#  helper to build targets
# ============================================================

def build_stardist_targets_routeB(
    mask_inst_np: np.ndarray,
    n_rays: int = 32,
    max_dist: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Targets for Route B:
      - objectness/foreground: (1,H,W) float32
      - distances per pixel : (R,H,W) float32

    Note: computing distances per pixel is expensive; consider precomputing offline.
    """
    fg = (mask_inst_np > 0).astype(np.float32)
    dists = compute_star_distances_per_pixel(mask_inst_np.astype(np.int32), n_rays=n_rays, max_dist=max_dist)

    fg_t = torch.from_numpy(fg).unsqueeze(0)              # (1,H,W)
    dists_t = torch.from_numpy(dists.astype(np.float32))  # (R,H,W)
    return fg_t, dists_t
