from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
from numba import njit, prange

# -----------------------------
# Config
# -----------------------------
N_RAYS = 64
MAX_DIST = 80
USE_MULTIPROC = True
N_WORKERS = 4

# -----------------------------
# Precompute ray offsets (integer grid walk via rounding of continuous ray)
# This matches your "round(y0 + step*dy)" behavior, but precomputed once.
# -----------------------------
def build_ray_offsets(n_rays: int, max_dist: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False).astype(np.float64)
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)

    offs = np.zeros((n_rays, max_dist, 2), dtype=np.int16)  # (ray, step-1, [dy,dx])
    for r in range(n_rays):
        dy = sin_a[r]
        dx = cos_a[r]
        for s in range(1, max_dist + 1):
            oy = int(np.round(s * dy))
            ox = int(np.round(s * dx))
            offs[r, s - 1, 0] = oy
            offs[r, s - 1, 1] = ox
    return offs

RAY_OFFSETS = build_ray_offsets(N_RAYS, MAX_DIST)

# -----------------------------
# Numba kernel
# -----------------------------
@njit(parallel=True, fastmath=True, cache=True)
def stardist_dists_from_offsets(mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    mask: (H,W) int32, 0=bg, >0 instance id
    offsets: (R,MAX_DIST,2) int16 [dy,dx]
    out: (R,H,W) float32
    """
    H, W = mask.shape
    R = offsets.shape[0]
    max_dist = offsets.shape[1]

    out = np.zeros((R, H, W), dtype=np.float32)

    for y0 in prange(H):
        for x0 in range(W):
            inst = mask[y0, x0]
            if inst == 0:
                continue

            for r in range(R):
                last_inside = 0
                for s in range(max_dist):
                    y = y0 + int(offsets[r, s, 0])
                    x = x0 + int(offsets[r, s, 1])
                    if y < 0 or y >= H or x < 0 or x >= W:
                        break
                    if mask[y, x] != inst:
                        break
                    last_inside = s + 1
                out[r, y0, x0] = last_inside

    return out

# -----------------------------
# Utils
# -----------------------------
def relabel_contiguous(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.int32, copy=False)
    ids = np.unique(mask)
    ids = ids[ids != 0]
    if ids.size == 0:
        return mask
    out = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_id in enumerate(ids, start=1):
        out[mask == old_id] = new_id
    return out

def precompute_one(mask_path: Path, out_path: Path) -> None:
    mask = np.load(mask_path).astype(np.int32, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be (H,W). Got {mask.shape} for {mask_path}")
    mask = relabel_contiguous(mask)

    d = stardist_dists_from_offsets(mask, RAY_OFFSETS)  # (R,H,W) float32
    np.save(out_path, d.astype(np.float32, copy=False))

def precompute_split(prep_root: Path, split: str) -> None:
    img_dir  = prep_root / split / "images"
    mask_dir = prep_root / split / "masks"
    out_dir  = prep_root / split / "targets_stardist"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("img_*.npy"))
    if not img_files:
        print(f"[WARN] No images in {img_dir}")
        return

    jobs = []
    for img_path in img_files:
        stem = img_path.stem.replace("img_", "")
        mask_path = mask_dir / f"mask_{stem}.npy"
        out_path  = out_dir  / f"dist_{stem}.npy"
        if out_path.exists():
            # sanity: si mauvais R, on force recalcul
            try:
                arr = np.load(out_path, mmap_mode="r")
                if arr.ndim == 3 and arr.shape[0] == N_RAYS:
                    continue
            except Exception:
                pass
        jobs.append((mask_path, out_path))

    if not jobs:
        return

    if USE_MULTIPROC:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(precompute_one, mp, op) for mp, op in jobs]
            for _ in tqdm(as_completed(futs), total=len(futs), desc=f"Precompute [{split}]"):
                pass
    else:
        for mask_path, out_path in tqdm(jobs, desc=f"Precompute [{split}]"):
            precompute_one(mask_path, out_path)

def main():
    base_dir = Path("../").resolve()
    prep_root = base_dir / "data" / "prepared" / "pannuke"
    print("prep_root:", prep_root)
    print(f"Config: N_RAYS={N_RAYS}, MAX_DIST={MAX_DIST}, multiproc={USE_MULTIPROC}, workers={N_WORKERS}")

    for split in ["train", "val", "test"]:
        if (prep_root / split).exists():
            precompute_split(prep_root, split)

    print("Done.")

if __name__ == "__main__":
    main()
