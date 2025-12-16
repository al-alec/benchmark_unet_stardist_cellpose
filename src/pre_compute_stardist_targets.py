from __future__ import annotations
from pathlib import Path
import numpy as np
from tqdm import tqdm

from models.stardist import compute_star_distances  # ROUTE A (centre par instance)

N_RAYS = 32

def precompute_split(prep_root: Path, split: str) -> None:
    img_dir  = prep_root / split / "images"
    mask_dir = prep_root / split / "masks"
    out_dir  = prep_root / split / "targets_stardist"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("img_*.npy"))
    if len(img_files) == 0:
        print(f"[WARN] No images in {img_dir}")
        return

    for img_path in tqdm(img_files, desc=f"Precompute StarDist {split}"):
        stem = img_path.stem.replace("img_", "")   # "000123"
        mask_path = mask_dir / f"mask_{stem}.npy"
        out_path  = out_dir  / f"dist_{stem}.npy"

        if out_path.exists():
            continue

        mask = np.load(mask_path).astype(np.int32)            # (H,W)
        dists = compute_star_distances(mask, n_rays=N_RAYS)    # (R,H,W) float32
        np.save(out_path, dists.astype(np.float32))

def main():
    base_dir = Path("../").resolve()
    prep_root = base_dir / "data" / "prepared" / "pannuke"

    print("prep_root:", prep_root)
    for split in ["train", "val", "test"]:
        if (prep_root / split).exists():
            precompute_split(prep_root, split)

    print("Done.")

if __name__ == "__main__":
    main()
