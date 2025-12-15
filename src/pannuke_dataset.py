from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List, Any, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def relabel_contiguous(mask: np.ndarray) -> np.ndarray:
    """
    Remap instance ids to contiguous {0,1..K}.
    Keeps 0 as background.
    """
    mask = mask.astype(np.int32, copy=False)
    ids = np.unique(mask)
    ids = ids[ids != 0]
    if ids.size == 0:
        return mask

    out = np.zeros_like(mask, dtype=np.int32)
    for new_id, old_id in enumerate(ids, start=1):
        out[mask == old_id] = new_id
    return out


def default_collate_pannuke(batch: List[Tuple[Tensor, Tensor, Optional[Tensor], str | None]]):
    """
    Batch elements:
      - img: (3,H,W) float32
      - mask: (H,W) int64
      - types: (H,W) int64 or None
      - name: str or None

    Returns:
      imgs: (B,3,H,W)
      masks: (B,H,W)
      types: (B,H,W) or None
      names: list[str] or None
    """
    imgs, masks, types_list, names = zip(*batch)

    imgs_b = torch.stack(imgs, dim=0)
    masks_b = torch.stack(masks, dim=0)

    if all(t is None for t in types_list):
        types_b = None
    else:
        # If some are None and some not, that's a dataset consistency bug.
        if any(t is None for t in types_list):
            raise RuntimeError("Inconsistent batch: some samples have types, others don't.")
        types_b = torch.stack(types_list, dim=0)

    if all(n is None for n in names):
        names_out = None
    else:
        names_out = list(names)

    return imgs_b, masks_b, types_b, names_out


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

@dataclass
class PannukeDatasetConfig:
    relabel_instances: bool = True
    validate_shapes: bool = False
    validate_ranges: bool = False  # checks image range [0,1] if you expect it


class PannukePreparedDataset(Dataset):
    """
    Dataset for prepared PanNuke:
      root/
        train|val|test/
          images/img_XXXXXX.npy   # (H,W,3) float32 in [0,1] (expected)
          masks/mask_XXXXXX.npy   # (H,W) int/uint, 0 bg, >0 instance ids
          types/type_XXXXXX.npy   # (H,W) uint8, 0 bg, >0 class id (optional)

    __getitem__ returns:
      img_t:  (3,H,W) float32
      mask_t: (H,W) int64
      types_t:(H,W) int64 or None
      (optionally) name
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[
            Callable[[Tensor, Tensor, Optional[Tensor]], Tuple[Tensor, Any, Optional[Tensor]]]
        ] = None,
        return_image_name: bool = False,
        cfg: Optional[PannukeDatasetConfig] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_image_name = return_image_name
        self.cfg = cfg or PannukeDatasetConfig()

        self.img_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.type_dir = self.root / split / "types"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing masks dir: {self.mask_dir}")

        self.img_files = sorted(self.img_dir.glob("img_*.npy"))
        if not self.img_files:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.has_types = self.type_dir.exists()

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        name = img_path.name

        mask_path = self.mask_dir / name.replace("img_", "mask_")
        type_path = self.type_dir / name.replace("img_", "type_")

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")

        img = np.load(img_path)  # expected (H,W,3)
        mask = np.load(mask_path)

        types = None
        if self.has_types and type_path.exists():
            types = np.load(type_path)

        # --- optional validations ---
        if self.cfg.validate_shapes:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Invalid image shape for {img_path}: {img.shape}")
            if mask.shape != img.shape[:2]:
                raise ValueError(f"Mask shape {mask.shape} != image spatial shape {img.shape[:2]} for {mask_path}")
            if types is not None and types.shape != mask.shape:
                raise ValueError(f"Types shape {types.shape} != mask shape {mask.shape} for {type_path}")

        if self.cfg.validate_ranges:
            if not np.issubdtype(img.dtype, np.floating):
                raise ValueError(f"Image {img_path} is not float dtype: {img.dtype}")
            if img.min() < -1e-3 or img.max() > 1.0 + 1e-3:
                raise ValueError(f"Image {img_path} seems out of [0,1] range: min={img.min()} max={img.max()}")

        # --- normalize instance ids (recommended) ---
        if self.cfg.relabel_instances:
            mask = relabel_contiguous(mask)

        # --- torch tensors ---
        img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # (3,H,W)
        mask_t = torch.from_numpy(mask.astype(np.int64))                   # (H,W)

        types_t: Optional[Tensor] = None
        if types is not None:
            types_t = torch.from_numpy(types.astype(np.int64))

        # --- transform (can change mask into dict targets, etc.) ---
        if self.transform is not None:
            img_t, mask_or_target, types_t = self.transform(img_t, mask_t, types_t)
        else:
            mask_or_target = mask_t

        if self.return_image_name:
            return img_t, mask_or_target, types_t, name

        return img_t, mask_or_target, types_t


# ------------------------------------------------------------
# Loader builders
# ------------------------------------------------------------

def build_pannuke_loaders(
    root: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 4,
    transform_train: Optional[Callable] = None,
    transform_eval: Optional[Callable] = None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
    return_image_name: bool = False,
    cfg: Optional[PannukeDatasetConfig] = None,
):
    root = Path(root)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_ds = PannukePreparedDataset(
        root=root,
        split="train",
        transform=transform_train,
        return_image_name=return_image_name,
        cfg=cfg,
    )
    val_ds = PannukePreparedDataset(
        root=root,
        split="val",
        transform=transform_eval,
        return_image_name=return_image_name,
        cfg=cfg,
    )

    try:
        test_ds = PannukePreparedDataset(
            root=root,
            split="test",
            transform=transform_eval,
            return_image_name=return_image_name,
            cfg=cfg,
        )
    except (FileNotFoundError, RuntimeError):
        test_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=default_collate_pannuke if return_image_name else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=default_collate_pannuke if return_image_name else None,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=default_collate_pannuke if return_image_name else None,
        )

    return train_loader, val_loader, test_loader
