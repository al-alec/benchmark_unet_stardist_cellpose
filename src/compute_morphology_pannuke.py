# compute_morphology_pannuke.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class MorphConfig:
    prepared_root: Path
    out_csv: Optional[Path] = None

    splits: List[str] = None

    # circularity can be noisy for tiny objects
    min_area_for_circularity: int = 30

    # whether to compute area quantiles globally for morph_class
    area_quantile_irregular: float = 0.99

    # thresholds for morph_class (you can tune later, but keep fixed for fairness)
    thr_round_circ: float = 0.80
    thr_round_ecc: float = 0.65
    thr_round_sol: float = 0.90

    thr_elong_ecc: float = 0.80
    thr_elong_sol: float = 0.85

    thr_irreg_circ: float = 0.55
    thr_irreg_sol: float = 0.80

    # extra checks
    warn_missing_types: bool = True
    warn_missing_instance_class: bool = True


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


def _compute_circularity(area: float, perimeter: float, min_area: int) -> float:
    if area < min_area:
        return 0.0
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _bbox_aspect(bbox) -> float:
    # bbox = (min_row, min_col, max_row, max_col)
    h = float(bbox[2] - bbox[0])
    w = float(bbox[3] - bbox[1])
    return _safe_div(w, h)  # width/height


def _load_optional(path: Path):
    if path.exists():
        return np.load(path)
    return None


def compute_pannuke_morphology(cfg: MorphConfig) -> pd.DataFrame:
    """
    Reads prepared PanNuke:
      prepared_root/split/images/img_*.npy
      prepared_root/split/masks/mask_*.npy
      prepared_root/split/types/type_*.npy                       (pixel class map)
      prepared_root/split/instances/instance_class_*.npy         (per-instance class map)

    Outputs a DataFrame with one row per GT nucleus instance.
    """
    root = Path(cfg.prepared_root)

    if cfg.splits is None:
        cfg.splits = ["train", "val", "test"]

    if cfg.out_csv is None:
        cfg.out_csv = root / "pannuke_morphology_with_classes.csv"
    else:
        cfg.out_csv = Path(cfg.out_csv)

    rows: List[Dict] = []

    for split in cfg.splits:
        img_dir = root / split / "images"
        mask_dir = root / split / "masks"
        type_dir = root / split / "types"
        inst_dir = root / split / "instances"

        if not img_dir.exists() or not mask_dir.exists():
            print(f"[WARN] Split '{split}' missing images/ or masks/. Skipping.")
            continue

        img_files = sorted(img_dir.glob("img_*.npy"))

        for img_path in tqdm(img_files, desc=f"Morphology {split}"):
            name = img_path.name
            mask_path = mask_dir / name.replace("img_", "mask_")
            type_path = type_dir / name.replace("img_", "type_")
            instmap_path = inst_dir / name.replace("img_", "instance_class_")

            if not mask_path.exists():
                print(f"[WARN] Missing mask for {name}, skipping.")
                continue

            mask = np.load(mask_path)  # (H,W) uint16/int, 0 bg

            # Optional: class info
            type_map = _load_optional(type_path)
            inst_class_map = _load_optional(instmap_path)

            if cfg.warn_missing_types and type_map is None:
                print(f"[WARN] Missing type map for {split}/{name} (types/).")
            if cfg.warn_missing_instance_class and inst_class_map is None:
                print(f"[WARN] Missing instance class map for {split}/{name} (instances/).")

            props = regionprops(mask)

            for p in props:
                inst_id = int(p.label)

                area = float(p.area)
                perimeter = float(p.perimeter or 0.0)

                ecc = float(p.eccentricity)
                sol = float(p.solidity)

                circ = _compute_circularity(area, perimeter, cfg.min_area_for_circularity)
                aspect = _bbox_aspect(p.bbox)

                # Get class_id: prefer per-instance mapping if available
                class_id = 0
                if inst_class_map is not None and inst_id < len(inst_class_map):
                    class_id = int(inst_class_map[inst_id])
                elif type_map is not None:
                    # majority vote from pixel-wise type_map inside instance
                    pix = (mask == inst_id)
                    cls_vals = type_map[pix]
                    cls_vals = cls_vals[cls_vals > 0]
                    if cls_vals.size > 0:
                        binc = np.bincount(cls_vals.astype(np.int32))
                        class_id = int(np.argmax(binc))

                rows.append(
                    {
                        "split": split,
                        "image": name,
                        "instance_id": inst_id,
                        "class_id": class_id,
                        "area": area,
                        "perimeter": perimeter,
                        "eccentricity": ecc,
                        "solidity": sol,
                        "circularity": circ,
                        "bbox_aspect": aspect,
                        "centroid_row": float(p.centroid[0]),
                        "centroid_col": float(p.centroid[1]),
                    }
                )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("[WARN] No instances found. Check your prepared dataset.")
        df.to_csv(cfg.out_csv, index=False)
        return df

    # ------------------------------------------------------------
    # Morphology class assignment (global thresholds + area Q)
    # ------------------------------------------------------------

    area_q = float(df["area"].quantile(cfg.area_quantile_irregular))

    def assign_morph(row):
        circ = row["circularity"]
        ecc = row["eccentricity"]
        sol = row["solidity"]
        area = row["area"]

        # 1) Round: convex, not elongated, fairly circular
        if (circ > cfg.thr_round_circ) and (ecc < cfg.thr_round_ecc) and (sol > cfg.thr_round_sol):
            return "round"

        # 2) Elongated: high eccentricity, still fairly solid
        if (ecc > cfg.thr_elong_ecc) and (sol > cfg.thr_elong_sol):
            return "elongated"

        # 3) Irregular: jagged/concave OR extremely large (potential artifacts/complex)
        if (circ < cfg.thr_irreg_circ) or (sol < cfg.thr_irreg_sol) or (area > area_q):
            return "irregular"

        return "other"

    df["morph_class"] = df.apply(assign_morph, axis=1)

    # Optional: size bins for later analysis (quantiles on GT)
    df["size_bin_q"] = pd.qcut(df["area"], q=3, labels=["small", "medium", "large"])

    # Save
    df.to_csv(cfg.out_csv, index=False)

    print(f"[OK] Saved: {cfg.out_csv} (rows={len(df)})")
    print("[INFO] morph_class distribution:")
    print(df["morph_class"].value_counts())
    print("[INFO] size_bin_q distribution:")
    print(df["size_bin_q"].value_counts())

    # Quick sanity: class_id distribution
    print("[INFO] class_id distribution (top 10):")
    print(df["class_id"].value_counts().head(10))

    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    prepared_root = base_dir / "data" / "prepared" / "pannuke"
    out_csv = prepared_root / "pannuke_morphology_with_classes.csv"

    cfg = MorphConfig(
        prepared_root=prepared_root,
        out_csv=out_csv,
        splits=["train", "val", "test"],
        min_area_for_circularity=30,
        area_quantile_irregular=0.99,
        warn_missing_types=True,
        warn_missing_instance_class=True,
    )

    print(f"Prepared root: {cfg.prepared_root}")
    print(f"Out CSV     : {cfg.out_csv}")

    compute_pannuke_morphology(cfg)
