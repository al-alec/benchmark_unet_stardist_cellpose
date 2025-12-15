# prepare_pannuke.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label as nd_label


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class PrepareConfig:
    pannuke_root: Path
    out_root: Path

    fold_to_split: Dict[str, str] = None

    keep_empty: bool = False
    use_8_connectivity: bool = True

    # Background channel handling
    assume_last_channel_is_background: bool = True
    background_channel_index: Optional[int] = None  # override if you know it

    # Diagnostics
    warn_overlaps: bool = True
    overlap_warn_ratio: float = 1e-6  # ratio of overlapped pixels vs all nucleus pixels

    # Output extras
    write_instance_class_csv: bool = True


def _mkdir_structure(out_root: Path, splits: List[str]) -> None:
    for split in splits:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)
        (out_root / split / "types").mkdir(parents=True, exist_ok=True)       # pixel-wise class map
        (out_root / split / "instances").mkdir(parents=True, exist_ok=True)   # per-instance class mapping


def _choose_bg_channel(mask_raw: np.ndarray, cfg: PrepareConfig) -> int:
    """
    Choose background channel index robustly if possible.
    mask_raw: (H,W,C)
    Returns bg_index
    """
    H, W, C = mask_raw.shape

    if cfg.background_channel_index is not None:
        if not (0 <= cfg.background_channel_index < C):
            raise ValueError(f"background_channel_index={cfg.background_channel_index} out of range for C={C}")
        return cfg.background_channel_index

    if cfg.assume_last_channel_is_background:
        return C - 1

    # Heuristic: background is the channel with the largest coverage where others are 0
    # This is imperfect but safer than guessing randomly.
    sums = mask_raw.reshape(-1, C).sum(axis=0)
    return int(np.argmax(sums))


def _label_instances_in_channel(binary: np.ndarray, use_8: bool) -> Tuple[np.ndarray, int]:
    """
    binary: (H,W) bool
    returns labeled (H,W) int32 and count n
    """
    if use_8:
        structure = np.ones((3, 3), dtype=np.int32)
    else:
        structure = None
    labeled, n = nd_label(binary, structure=structure)
    return labeled.astype(np.int32, copy=False), int(n)


def prepare_pannuke_instances(cfg: PrepareConfig) -> None:
    """
    Prepares PanNuke into:
      out_root/split/images/img_XXXXXX.npy        (H,W,3) float32 in [0,1]
      out_root/split/masks/mask_XXXXXX.npy        (H,W) uint16 instance ids, 0 bg
      out_root/split/types/type_XXXXXX.npy        (H,W) uint8 class ids, 0 bg, 1..Kclasses
      out_root/split/instances/instance_class_XXXXXX.npy  (K_inst+1,) uint8 mapping:
           map[instance_id] = class_id (0 for background / unknown)

    Also optionally writes: out_root/instance_class_map.csv
    """
    pannuke_root = Path(cfg.pannuke_root)
    out_root = Path(cfg.out_root)

    if cfg.fold_to_split is None:
        cfg.fold_to_split = {
            "fold1": "train",
            "fold2": "val",
            "fold3": "test",
        }

    if not pannuke_root.exists():
        raise FileNotFoundError(f"Source root not found: {pannuke_root}")

    splits = sorted(set(cfg.fold_to_split.values()))
    _mkdir_structure(out_root, splits)

    global_idx = 0
    csv_rows = []

    for fold_name, split in cfg.fold_to_split.items():
        fold_dir = pannuke_root / fold_name
        images_npy = fold_dir / "images.npy"
        masks_npy = fold_dir / "masks.npy"

        if not images_npy.exists() or not masks_npy.exists():
            print(f"[WARN] Missing files in {fold_dir}, skipping.")
            continue

        images = np.load(images_npy)  # (N,H,W,3) uint8 typically
        masks = np.load(masks_npy)    # (N,H,W,C) 0/1 or 0..?

        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Unexpected images shape in {images_npy}: {images.shape}")
        if masks.ndim != 4:
            raise ValueError(f"Unexpected masks shape in {masks_npy}: {masks.shape}")
        if masks.shape[0] != images.shape[0] or masks.shape[1:3] != images.shape[1:3]:
            raise ValueError("images and masks spatial dimensions do not match.")

        split_img_dir = out_root / split / "images"
        split_mask_dir = out_root / split / "masks"
        split_type_dir = out_root / split / "types"
        split_inst_dir = out_root / split / "instances"

        for i in tqdm(range(images.shape[0]), desc=f"{fold_name} -> {split}"):
            img = images[i]          # (H,W,3)
            mask_raw = masks[i]      # (H,W,C)

            # Skip empty (optional)
            if mask_raw.sum() == 0:
                if cfg.keep_empty:
                    # still save empty sample (all zeros masks/types)
                    pass
                else:
                    continue

            # Normalize image to float32 [0,1]
            img_f = img.astype(np.float32) / 255.0

            H, W, C = mask_raw.shape
            bg_idx = _choose_bg_channel(mask_raw, cfg)

            # We'll treat all channels except bg as nucleus classes
            class_channels = [c for c in range(C) if c != bg_idx]
            # We assign class_id in 1..len(class_channels) by channel order,
            # but to keep consistent with original channel indices, we can use (c+1)
            # Here: use class_id = c+1, excluding bg channel.
            # That keeps mapping stable across files.
            # Example: if bg is last, classes are 1..C-1 as expected.

            mask_instance = np.zeros((H, W), dtype=np.uint16)
            type_map = np.zeros((H, W), dtype=np.uint8)  # pixel-wise class id
            current_label = 1

            # For overlap diagnostics
            nucleus_pixels_total = 0
            overlap_pixels_total = 0

            # Build instance map + type map
            # Note: if overlaps occur, we decide to keep first-come or override. Here we keep highest "class priority"
            # by score = channel order. But since mask_raw channels are binary, simplest is "do not override existing instance pixels".
            for c in class_channels:
                binary = mask_raw[..., c] > 0
                if not np.any(binary):
                    continue

                labeled, n = _label_instances_in_channel(binary, cfg.use_8_connectivity)
                if n == 0:
                    continue

                # shift ids to be globally unique within image
                nz = labeled > 0
                labeled[nz] += (current_label - 1)

                # overlap check: where we already have an instance and new labeled is nonzero
                overlap = (mask_instance > 0) & nz
                if overlap.any():
                    overlap_pixels_total += int(overlap.sum())
                nucleus_pixels_total += int(nz.sum())

                # IMPORTANT policy: do not overwrite existing instances pixels
                write_mask = nz & (mask_instance == 0)
                mask_instance[write_mask] = labeled[write_mask].astype(np.uint16)

                # type_map should match the class of the pixel
                # class_id = original channel index + 1, except bg channel is ignored
                class_id = int(c + 1)
                type_map[write_mask] = np.uint8(class_id)

                current_label += n

            # Warn on overlaps if requested
            if cfg.warn_overlaps and nucleus_pixels_total > 0:
                ratio = overlap_pixels_total / float(nucleus_pixels_total)
                if ratio > cfg.overlap_warn_ratio:
                    print(
                        f"[WARN] Overlap detected in {fold_name}/{i}: "
                        f"overlap_pixels={overlap_pixels_total} nucleus_pixels={nucleus_pixels_total} ratio={ratio:.2e}"
                    )

            # If keep_empty is True and this was empty, mask_instance and type_map are already zeros.
            # Relabel contiguous 1..K (optional but recommended)
            # We'll build instance_id -> class_id mapping via majority vote on type_map
            ids = np.unique(mask_instance)
            ids = ids[ids > 0]

            if ids.size > 0:
                # Relabel to contiguous
                relabeled = np.zeros_like(mask_instance, dtype=np.uint16)
                inst_class_map = np.zeros((int(ids.size) + 1,), dtype=np.uint8)  # index 0 reserved
                for new_id, old_id in enumerate(ids, start=1):
                    pix = (mask_instance == old_id)
                    relabeled[pix] = np.uint16(new_id)

                    # majority vote for class id among pixels
                    cls_vals = type_map[pix]
                    if cls_vals.size == 0:
                        inst_class_map[new_id] = 0
                    else:
                        # ignore zeros just in case
                        cls_vals = cls_vals[cls_vals > 0]
                        if cls_vals.size == 0:
                            inst_class_map[new_id] = 0
                        else:
                            # majority vote
                            binc = np.bincount(cls_vals.astype(np.int32))
                            inst_class_map[new_id] = np.uint8(int(np.argmax(binc)))

                    if cfg.write_instance_class_csv:
                        csv_rows.append(
                            {
                                "split": split,
                                "img_index_global": global_idx + 1,
                                "image_name": None,  # filled after naming
                                "instance_id": int(new_id),
                                "class_id": int(inst_class_map[new_id]),
                            }
                        )

                mask_instance = relabeled
            else:
                inst_class_map = np.zeros((1,), dtype=np.uint8)

            # Save
            global_idx += 1
            img_name = f"img_{global_idx:06d}.npy"
            mask_name = f"mask_{global_idx:06d}.npy"
            type_name = f"type_{global_idx:06d}.npy"
            instmap_name = f"instance_class_{global_idx:06d}.npy"

            np.save(split_img_dir / img_name, img_f)
            np.save(split_mask_dir / mask_name, mask_instance)
            np.save(split_type_dir / type_name, type_map)
            np.save(split_inst_dir / instmap_name, inst_class_map)

            if cfg.write_instance_class_csv and ids.size > 0:
                # fill image_name for rows added in this image
                # rows were appended in order; update last K rows
                K = int(ids.size)
                for j in range(1, K + 1):
                    csv_rows[-j]["image_name"] = img_name

    if cfg.write_instance_class_csv:
        df = pd.DataFrame(csv_rows)
        csv_path = out_root / "instance_class_map.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] Wrote instance-class mapping CSV: {csv_path} ({len(df)} rows)")

    print("[OK] PanNuke preparation completed.")


if __name__ == "__main__":
    base_dir = Path("../")
    pannuke_root = base_dir / "data" / "pannuke"
    out_root = base_dir / "data" / "prepared" / "pannuke"

    cfg = PrepareConfig(
        pannuke_root=pannuke_root,
        out_root=out_root,
        keep_empty=False,             # set True if you want background-only samples
        use_8_connectivity=True,
        assume_last_channel_is_background=True,
        background_channel_index=None,
        warn_overlaps=True,
        overlap_warn_ratio=1e-6,
        write_instance_class_csv=True,
    )

    print(f"PanNuke root : {cfg.pannuke_root}")
    print(f"Output root  : {cfg.out_root}")

    prepare_pannuke_instances(cfg)
