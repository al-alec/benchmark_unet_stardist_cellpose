from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class PannukePreparedDataset(Dataset):
    """
    Dataset pour PanNuke préparé au format canonique :

    root/
      train/
        images/img_XXXXXX.npy  # (H, W, 3), float32 dans [0,1]
        masks/mask_XXXXXX.npy  # (H, W), uint16/int, 0 = fond, >0 = instances
      val/
      test/

    __getitem__ renvoie (image, mask_instance) :
      - image : Tensor (3, H, W), float32
      - mask  : Tensor (H, W), int64
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
        return_image_name: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_image_name = return_image_name

        img_dir = self.root / split / "images"
        if not img_dir.exists():
            raise FileNotFoundError(f"Dossier images introuvable : {img_dir}")

        self.img_files = sorted(img_dir.glob("img_*.npy"))
        if len(self.img_files) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {img_dir}")

        mask_dir = self.root / split / "masks"
        if not mask_dir.exists():
            raise FileNotFoundError(f"Dossier masks introuvable : {mask_dir}")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        mask_path = (
            self.root / self.split / "masks" /
            img_path.name.replace("img_", "mask_")
        )

        img = np.load(img_path)   # (H, W, 3)
        mask = np.load(mask_path) # (H, W)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Image {img_path} a une shape inattendue : {img.shape}")
        if mask.shape != img.shape[:2]:
            raise ValueError(
                f"Mask {mask_path} shape {mask.shape} != image {img.shape[:2]}"
            )

        img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # (3,H,W)
        mask_t = torch.from_numpy(mask.astype(np.int64))                   # (H,W)

        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t)

        if self.return_image_name:
            return img_t, mask_t, img_path.name

        return img_t, mask_t


def build_pannuke_loaders(
    root: str | Path,
    batch_size: int = 4,
    num_workers: int = 4,
    transform_train: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
    transform_eval: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Construit les DataLoader train / val / test à partir du dossier préparé.

    root doit pointer vers : .../data/prepared/pannuke
    """

    root = Path(root)

    train_ds = PannukePreparedDataset(root, split="train", transform=transform_train)
    val_ds   = PannukePreparedDataset(root, split="val",   transform=transform_eval)

    try:
        test_ds = PannukePreparedDataset(root, split="test", transform=transform_eval)
    except (FileNotFoundError, RuntimeError):
        test_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
