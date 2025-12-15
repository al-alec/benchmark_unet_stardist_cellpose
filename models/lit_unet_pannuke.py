from pathlib import Path

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import albumentations as A
# from albumentations.pytorch import ToTensorV2
import wandb

from src.pannuke_dataset import PannukePreparedDataset
from unet import UNet
from src.losses import bce_dice_loss

torch.set_float32_matmul_precision('high')

BASE_DIR = Path(
    "../"
)
DATA_ROOT = BASE_DIR / "data" / "prepared" / "pannuke"
# CKPT_DIR = BASE_DIR / "checkpoints"
# CKPT_DIR.mkdir(parents=True, exist_ok=True)


# def prob_to_instances(prob, thr=0.5, min_size=10):
#     # prob: (H,W) en [0,1]
#     bin_mask = prob > thr
#
#     #  nettoyage de bruit
#     bin_mask = remove_small_objects(bin_mask, min_size=min_size)
#
#     # distance transform (plus grand au centre des blobs)
#     dist = ndi.distance_transform_edt(bin_mask)
#
#     # seeds: pics locaux dans le dist
#     local_max = (dist == ndi.maximum_filter(dist, size=5))
#     markers, _ = ndi.label(local_max)
#
#     # watershed
#     labels_ws = watershed(-dist, markers, mask=bin_mask)
#
#     return labels_ws.astype(np.int32)

# def bin_to_instances(pred_prob: np.ndarray, thr=0.5):
#     return prob_to_instances(pred_prob, thr=thr, min_size=10)

def simple_transform(img, mask, types_t=None):
    """
    img : (3,H,W) float32
    mask : (H,W) int64 (ids instances)

    On renvoie :
      - img inchangé
      - mask_bin : (1,H,W) float32, 1 = noyau, 0 = fond
    """
    mask_bin = (mask > 0).float().unsqueeze(0)
    return img, mask_bin, types_t


train_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            # shear laissé par défaut (0,0)
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.3,
        ),
    ]
)



def train_transform(img_t, mask_t, types_t=None):
    img = img_t.permute(1, 2, 0).cpu().numpy()          # (H,W,3)
    mask = mask_t.cpu().numpy().astype(np.int32)        # (H,W)

    aug = train_aug(image=img, mask=mask)
    img_aug = aug["image"]
    mask_aug = aug["mask"].astype(np.int32)

    img_aug_t = torch.from_numpy(img_aug).permute(2, 0, 1).float()
    mask_aug_t = torch.from_numpy(mask_aug).long()

    mask_bin = (mask_aug_t > 0).float().unsqueeze(0)
    return img_aug_t, mask_bin, types_t




class PannukeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        # stage peut être "fit", "validate", "test", "predict" ou None
        if stage in (None, "fit"):
            self.train_ds = PannukePreparedDataset(
                root=self.root,
                split="train",
                transform=train_transform,
            )
            self.val_ds = PannukePreparedDataset(
                root=self.root,
                split="val",
                transform=simple_transform,
            )

        if stage in (None, "test"):
            try:
                self.test_ds = PannukePreparedDataset(
                    root=self.root,
                    split="test",
                    transform=simple_transform,
                )
            except (FileNotFoundError, RuntimeError):
                self.test_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class UNetLightning(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, weight_decay: float = 1e-4, base_ch: int = 64):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels=3, out_channels=1, base_ch=base_ch)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask, _ = batch   # img: (B,3,H,W), mask: (B,1,H,W)
        logits = self(img)
        loss = bce_dice_loss(logits, mask)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, _ = batch
        logits = self(img)
        loss = bce_dice_loss(logits, mask)
        self.log("val_loss",  loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


def main():
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # DataModule
    dm = PannukeDataModule(
        root=DATA_ROOT,
        batch_size=4,
        num_workers=4,
    )

    # Modèle Lightning
    model = UNetLightning(lr=1e-4, weight_decay=1e-4, base_ch=64)

    # Logger W&B
    wandb_logger = WandbLogger(
        project="segmentation_pannuke",
        name="unet_lightning_bce_dice",
        log_model=True,
    )

    # Checkpoint du meilleur modèle (sur val_loss)
    ckpt_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="unet_pannuke_lit_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        min_delta=1e-4,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=device,
        devices=1,
        logger=wandb_logger,
        callbacks=[ckpt_callback, early_stop, TQDMProgressBar(refresh_rate=1),],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm,
                # ckpt_path="checkpoints/last-v15.ckpt"
                )

    # test sur le split "test"
    if dm.test_dataloader() is not None:
        trainer.test(model, datamodule=dm, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    main()
