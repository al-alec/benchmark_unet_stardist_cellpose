# lit_stardist_pannuke.py
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import ioumatch
import numpy as np
import torch
from torch.utils.data import DataLoader

import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.losses import stardist_loss

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------

HERE = Path.cwd().resolve()
PROJECT_ROOT = HERE.parent
MODELS_DIR = PROJECT_ROOT / "models"
CKPT_DIR = MODELS_DIR / "checkpoints"

if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))

BASE_DIR = Path("../")
DATA_ROOT = BASE_DIR / "data" / "prepared" / "pannuke"

# --------------------------------------------------------------------
# Imports project
# --------------------------------------------------------------------

from src.pannuke_dataset import PannukePreparedDataset
from models.stardist import (
    StarDist,
    build_stardist_targets_routeB,
    stardist_decode,
)

torch.set_float32_matmul_precision("high")

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

N_RAYS = 32

BATCH_SIZE = 8
NUM_WORKERS = 4

LR = 1e-4
BASE_CH = 64

# Loss balancing
W_PROB = 1.0
W_DIST = 0.2

# Stability (PanNuke can have big objects)
CLAMP_DIST = 80.0          # pixels
NORMALIZE_DIST = False
NORMALIZE_DIV = 64.0

# Decode
PROB_THR = 0.5
NMS_IOU_THR = 0.3
MIN_AREA = 10
MAX_CANDIDATES = 3000

# Sanity
SANITY_EVERY_N_EPOCHS = 2
SANITY_IOU_THR = 0.5
SANITY_IMAGE_INDEX = 0     # which sample to probe in callback

WANDB_PROJECT = "segmentation_pannuke"
RUN_NAME = "stardist_routeB_lightning"


# --------------------------------------------------------------------
# Albumentations (image + instance mask)
# --------------------------------------------------------------------

train_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
    ]
)


def _to_numpy_image(img_t: torch.Tensor) -> np.ndarray:
    """(3,H,W) torch -> (H,W,3) numpy"""
    return img_t.permute(1, 2, 0).detach().cpu().numpy()


def _to_numpy_mask(mask_t: torch.Tensor) -> np.ndarray:
    """(H,W) torch -> (H,W) int32 numpy"""
    return mask_t.detach().cpu().numpy().astype(np.int32)


def stardist_train_transform(
    img_t: torch.Tensor,
    mask_t: torch.Tensor,
    types_t: torch.Tensor | None = None,
):
    """
    Train transform:
      - apply augmentations jointly on image + instance mask
      - build Route B targets (foreground + distances per pixel)
    """
    img = _to_numpy_image(img_t)
    mask = _to_numpy_mask(mask_t)

    aug = train_aug(image=img, mask=mask)
    img_aug = aug["image"]
    mask_aug = aug["mask"].astype(np.int32)

    img_aug_t = torch.from_numpy(img_aug).permute(2, 0, 1).float()

    fg_t, dists_t = build_stardist_targets_routeB(
        mask_inst_np=mask_aug,
        n_rays=N_RAYS,
        max_dist=None,   # can set to e.g. 128 to cap compute
    )

    target = {
        "foreground": fg_t,   # (1,H,W)
        "distances": dists_t  # (R,H,W)
    }
    return img_aug_t, target, types_t


def stardist_eval_transform(
    img_t: torch.Tensor,
    mask_t: torch.Tensor,
    types_t: torch.Tensor | None = None,
):
    """
    Eval transform: no aug, build Route B targets.
    """
    mask = _to_numpy_mask(mask_t)

    fg_t, dists_t = build_stardist_targets_routeB(
        mask_inst_np=mask,
        n_rays=N_RAYS,
        max_dist=None,
    )

    target = {
        "foreground": fg_t,
        "distances": dists_t,
    }
    return img_t, target, types_t


# --------------------------------------------------------------------
# DataModule
# --------------------------------------------------------------------

class PannukeStarDistDataModule(pl.LightningDataModule):
    def __init__(self, root: str | Path, batch_size: int, num_workers: int):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_ds = PannukePreparedDataset(
                root=self.root,
                split="train",
                transform=stardist_train_transform,
            )
            self.val_ds = PannukePreparedDataset(
                root=self.root,
                split="val",
                transform=stardist_eval_transform,
            )

        if stage in (None, "test"):
            try:
                self.test_ds = PannukePreparedDataset(
                    root=self.root,
                    split="test",
                    transform=stardist_eval_transform,
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
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
        )


# --------------------------------------------------------------------
# LightningModule
# --------------------------------------------------------------------

class StarDistLightning(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        base_ch: int,
        n_rays: int,
        w_prob: float,
        w_dist: float,
        clamp_dist: float | None = None,
        normalize_dist: bool = False,
        normalize_div: float = 64.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = StarDist(input_ch=3, n_rays=n_rays, base_ch=base_ch)

        self.w_prob = w_prob
        self.w_dist = w_dist

        self.clamp_dist = clamp_dist
        self.normalize_dist = normalize_dist
        self.normalize_div = normalize_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _build_target_tensor(target_dict: dict) -> torch.Tensor:
        """
        target_dict:
          - "foreground": (B,1,H,W)
          - "distances" : (B,R,H,W)
        return:
          - (B,1+R,H,W)
        """
        fg = target_dict["foreground"]
        dist = target_dict["distances"]
        return torch.cat([fg, dist], dim=1)

    def _shared_step(self, batch, stage: str):
        imgs, target_dict, _ = batch
        preds = self(imgs)

        targets = self._build_target_tensor(target_dict).type_as(preds)

        loss = stardist_loss(
            preds,
            targets,
            w_prob=self.w_prob,
            w_dist=self.w_dist,
            clamp_dist=self.clamp_dist,
            normalize_dist=self.normalize_dist,
            normalize_div=self.normalize_div,
            # use_focal=True, focal_alpha=0.25, focal_gamma=2.0,  # optionally
            dist_loss="smooth_l1",
            smooth_l1_beta=1.0,
        )

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=imgs.size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )


# --------------------------------------------------------------------
# Sanity callback (decode + IoU matching)
# --------------------------------------------------------------------

class StarDistSanityCallback(pl.Callback):
    def __init__(
        self,
        data_root: Path,
        split: str,
        every_n_epochs: int,
        prob_thr: float,
        nms_iou_thr: float,
        min_area: int,
        max_candidates: int,
        iou_thr: float,
        sample_index: int = 0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.every_n_epochs = every_n_epochs

        self.prob_thr = prob_thr
        self.nms_iou_thr = nms_iou_thr
        self.min_area = min_area
        self.max_candidates = max_candidates
        self.iou_thr = iou_thr
        self.sample_index = sample_index

        self.ds = PannukePreparedDataset(
            root=self.data_root,
            split=self.split,
            transform=None,
            return_image_name=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        device = pl_module.device

        img_t, mask_t, _, name = self.ds[self.sample_index]
        img_b = img_t.unsqueeze(0).to(device)

        gt_inst = mask_t.detach().cpu().numpy().astype(np.int32)

        with torch.no_grad():
            out = pl_module(img_b)  # (1,1+R,H,W)

        prob_logits = out[0, 0].detach().cpu().numpy()
        prob_map = 1.0 / (1.0 + np.exp(-prob_logits))

        dist_map = out[0, 1:].detach().cpu().numpy().astype(np.float32)  # already softplus in model
        dist_map = np.maximum(dist_map, 0.0)

        pred_inst = stardist_decode(
            prob_map=prob_map,
            dist_map=dist_map,
            prob_thr=self.prob_thr,
            nms_iou_thr=self.nms_iou_thr,
            use_local_maxima=True,
            local_max_footprint=3,
            max_candidates=self.max_candidates,
            min_area=self.min_area,
        )

        res = ioumatch.evaluate_image(
            pred_inst.astype(np.int32),
            gt_inst.astype(np.int32),
            threshold=self.iou_thr,
            method="greedy",
            inclusive=False,
            normalize=False,
        )
        M = res["iou_matrix"]
        nb_matches = int((M >= self.iou_thr).sum())

        print(
            f"[SANITY EPOCH {epoch}] image={name} "
            f"IoU shape={M.shape} matches≥{self.iou_thr}: {nb_matches}"
        )

        if trainer.logger is not None:
            trainer.logger.experiment.log(
                {
                    "sanity/epoch": epoch,
                    "sanity/nb_matches": nb_matches,
                    "sanity/iou_matrix_size": int(M.size),
                    "sanity/nb_pred_instances": int(pred_inst.max()),
                    "sanity/nb_gt_instances": int(gt_inst.max()),
                }
            )

        pl_module.train()


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------

def main():
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator:", accelerator)

    dm = PannukeStarDistDataModule(
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = StarDistLightning(
        lr=LR,
        base_ch=BASE_CH,
        n_rays=N_RAYS,
        w_prob=W_PROB,
        w_dist=W_DIST,
        clamp_dist=CLAMP_DIST,
        normalize_dist=NORMALIZE_DIST,
        normalize_div=NORMALIZE_DIV,
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        log_model=True,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_callback = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="stardist_pannuke_routeB_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        min_delta=1e-4,
    )

    sanity_cb = StarDistSanityCallback(
        data_root=DATA_ROOT,
        split="val",
        every_n_epochs=SANITY_EVERY_N_EPOCHS,
        prob_thr=PROB_THR,
        nms_iou_thr=NMS_IOU_THR,
        min_area=MIN_AREA,
        max_candidates=MAX_CANDIDATES,
        iou_thr=SANITY_IOU_THR,
        sample_index=SANITY_IMAGE_INDEX,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=1,
        logger=wandb_logger,
        callbacks=[ckpt_callback, early_stop, TQDMProgressBar(refresh_rate=1), sanity_cb],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)

    if dm.test_dataloader() is not None:
        trainer.test(model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    main()
