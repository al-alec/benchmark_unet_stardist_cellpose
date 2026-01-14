# lit_stardist_pannuke.py
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

import ioumatch

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

torch.set_float32_matmul_precision("high")


from src.pannuke_dataset import PannukePreparedDataset


from models.stardist import StarDist as StarDistModel, pick_candidates, is_prob_map_flat
from models.stardist import stardist_losses as stardist_losses_fn
from models.stardist import stardist_decode as stardist_decode_fn


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

N_RAYS = 64

# PanNuke: types: 0=bg, 1..5=cell types
N_NUCLEUS_CLASSES = 5          # sans background
N_CLASSES_WITH_BG = N_NUCLEUS_CLASSES + 1

BATCH_SIZE = 4
NUM_WORKERS = 4

POS_WEIGHT  = 10.0

LR = 1e-4
BASE_CH = 32

# Loss balancing (paper style: BCE + MAE + CE + Tversky)
W_PROB = 1.0
W_DIST = 1.0
W_CE = 0.3
W_TVERSKY = 0.3
TV_ALPHA = 0.3
TV_BETA = 0.7
TV_INCLUDE_BG = True
LOCAL_MAX_FOOTPRINT = 11

# Stability / target scaling
CLAMP_DIST = 80.0          # pixels
# NORMALIZE_DIST = True
NORMALIZE_DIST = False
NORMALIZE_DIV = 1.0

# Decode
PROB_THR = 0.35
NMS_IOU_THR = 0.3
MIN_AREA = 10
MAX_CANDIDATES = 500
VOTE_THR = 0.5

# Sanity
SANITY_EVERY_N_EPOCHS = 1
SANITY_IOU_THR = 0.5
SANITY_IMAGE_INDEX = 0

WANDB_PROJECT = "segmentation_pannuke"
RUN_NAME = "stardist_lightning"

# --------------------------------------------------------------------
# Albumentations (photometric only here, no spatial)
# --------------------------------------------------------------------

train_aug = A.Compose(
    [
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
    ]
)

def _to_numpy_image(img_t: torch.Tensor) -> np.ndarray:
    # (3,H,W) -> (H,W,3)
    return img_t.permute(1, 2, 0).detach().cpu().numpy()

def stardist_train_transform(img_t, mask_t, types_t=None, dist_t=None):
    """
    Returns:
      img_aug_t: (3,H,W) float
      target: dict with keys:
        - foreground: (1,H,W) float
        - distances : (R,H,W) float
        - classes   : (H,W) long in 0..C (bg included, class0=bg)
    """
    if dist_t is None:
        raise RuntimeError("dist_t is None. Use load_dists=True in dataset.")

    img = _to_numpy_image(img_t)
    aug = train_aug(image=img)
    img_aug_t = torch.from_numpy(aug["image"]).permute(2, 0, 1).float()

    fg_t = (mask_t > 0).float().unsqueeze(0)  # (1,H,W)

    d = dist_t.float()
    if CLAMP_DIST is not None:
        d = torch.clamp(d, 0.0, float(CLAMP_DIST))
    # if NORMALIZE_DIST:
    #     d = d / float(NORMALIZE_DIV)

    target = {"foreground": fg_t, "distances": d}

    # classes: 0=bg, 1..N_NUCLEUS_CLASSES
    if types_t is not None:
        cls = types_t.long()
        # clamp to [0..N_NUCLEUS_CLASSES] to avoid garbage labels
        cls = torch.clamp(cls, 0, int(N_NUCLEUS_CLASSES))
        target["classes"] = cls

    return img_aug_t, target, types_t

def stardist_eval_transform(img_t, mask_t, types_t=None, dist_t=None):
    if dist_t is None:
        raise RuntimeError("dist_t is None. Use load_dists=True in dataset.")

    fg_t = (mask_t > 0).float().unsqueeze(0)

    d = dist_t.float()
    if CLAMP_DIST is not None:
        d = torch.clamp(d, 0.0, float(CLAMP_DIST))
    # if NORMALIZE_DIST:
    #     d = d / float(NORMALIZE_DIV)

    target = {"foreground": fg_t, "distances": d}

    if types_t is not None:
        cls = torch.clamp(types_t.long(), 0, int(N_NUCLEUS_CLASSES))
        target["classes"] = cls

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
                load_dists=True,
                n_rays=N_RAYS,
            )
            self.val_ds = PannukePreparedDataset(
                root=self.root,
                split="val",
                transform=stardist_eval_transform,
                load_dists=True,
                n_rays=N_RAYS,
            )

        if stage in (None, "test"):
            try:
                self.test_ds = PannukePreparedDataset(
                    root=self.root,
                    split="test",
                    transform=stardist_eval_transform,
                    load_dists=True,
                    n_rays=N_RAYS,
                )
            except (FileNotFoundError, RuntimeError):
                self.test_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # <- Windows friendly
            pin_memory=False,  # <- évite pin thread + copies
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
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
        n_nucleus_classes: int,
        w_prob: float,
        w_dist: float,
        w_ce: float,
        w_tversky: float,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_include_background: bool = True,
        normalize_dist: bool = False,
        normalize_div: float = 64.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = StarDistModel(
            input_ch=3,
            n_rays=n_rays,
            n_nucleus_classes=n_nucleus_classes,
            base_ch=base_ch,
            max_dist=CLAMP_DIST,
        )

        self.normalize_dist = normalize_dist
        self.normalize_div = normalize_div

    def forward(self, x: torch.Tensor):
        return self.model(x)  # prob_logits, dist_pos, class_logits

    def _shared_step(self, batch, stage: str):
        imgs, target_dict, _ = batch

        fg_t = target_dict["foreground"].float()      # (B,1,H,W)
        dists_t = target_dict["distances"].float()    # (B,R,H,W)
        cls_t = target_dict.get("classes", None)      # (B,H,W) in 0..C (bg included) or None

        prob_logits, dist_pos, class_logits = self(imgs)

        # si on normalises les targets, normalise aussi la sortie
        # if self.normalize_dist:
        #     dist_pos = dist_pos / float(self.normalize_div)

        losses = stardist_losses_fn(
            prob_logits=prob_logits,
            dist_pos=dist_pos,
            class_logits=class_logits,
            fg_t=fg_t.type_as(prob_logits),
            dist_t=dists_t.type_as(dist_pos),
            cls_t=cls_t,
            w_prob=self.hparams.w_prob,
            w_dist=self.hparams.w_dist,
            w_ce=self.hparams.w_ce,
            w_tversky=self.hparams.w_tversky,
            tversky_alpha=self.hparams.tversky_alpha,
            tversky_beta=self.hparams.tversky_beta,
            tversky_include_background=self.hparams.tversky_include_background,
            pos_weight=POS_WEIGHT,
        )

        total = losses["total"]
        self.log(f"{stage}_loss", total, prog_bar=True, on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log(f"{stage}_loss_prob", losses["prob"], on_step=False, on_epoch=True, batch_size=imgs.size(0))
        self.log(f"{stage}_loss_dist", losses["dist"], on_step=False, on_epoch=True, batch_size=imgs.size(0))

        # si pas de cls_t, ces valeurs peuvent être 0.0
        if "ce" in losses:
            self.log(f"{stage}_loss_ce", losses["ce"], on_step=False, on_epoch=True, batch_size=imgs.size(0))
        if "tversky" in losses:
            self.log(f"{stage}_loss_tversky", losses["tversky"], on_step=False, on_epoch=True, batch_size=imgs.size(0))

        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)


# --------------------------------------------------------------------
# Sanity callback (decode + IoU matching)
# --------------------------------------------------------------------

def _prob_peak(prob_map: np.ndarray, dist_map: np.ndarray) -> np.ndarray:
    """Même logique que toi: prob * mean_dist_normalized"""
    center = dist_map.mean(axis=0)
    center = center / (float(center.max()) + 1e-6)
    return prob_map * center


def _auto_thr(prob_in: np.ndarray, base_thr: float) -> tuple[float, float]:
    """Même logique que toi: clamp par peak_max"""
    peak_max = float(prob_in.max())
    thr = min(float(base_thr), 0.9 * peak_max)
    thr = max(thr, 0.05)
    return thr, peak_max


def _gt_seedmap_from_gt_dist(gt_inst: np.ndarray, gt_dist: np.ndarray) -> np.ndarray:
    """
    EXACTEMENT 1 seed par instance:
    seed = pixel avec max(mean_ray_dist) dans l'instance.
    """
    mean_dist = gt_dist.mean(axis=0).astype(np.float32, copy=False)
    seed = np.zeros_like(mean_dist, dtype=np.float32)
    n = int(gt_inst.max())
    for k in range(1, n + 1):
        m = (gt_inst == k)
        if not m.any():
            continue
        ys, xs = np.where(m)
        vals = mean_dist[ys, xs]
        j = int(vals.argmax())
        y0, x0 = int(ys[j]), int(xs[j])
        seed[y0, x0] = 1.0
    return seed


class StarDistSanityCallback(pl.Callback):
    """
    Diagnostic robuste:
      - GTSeed + GTDist doit reconstruire gt_n (HARD sanity dataset/dist/decoder)
      - Decode "policy" unifiée: prob_peak + auto_thr + pick_candidates (dans decoder)
      - Log: prob flat stats, peak_max, thr_final, nb_candidates, nb_pred, explosion
      - Ablations: PredProb+GTDist, GTSeed+PredDist, GTSeed+GTDist, Real
    """

    def __init__(
        self,
        data_root,
        split,
        every_n_epochs,
        prob_thr,
        nms_iou_thr,
        min_area,
        max_candidates,
        vote_thr,
        iou_thr,
        sample_index=0,
        local_max_footprint=11,
        explode_factor=10.0,
        early_vote_thr=0.8,
        early_max_candidates=200,
    ):
        super().__init__()
        from src.pannuke_dataset import PannukePreparedDataset

        self.data_root = data_root
        self.split = split
        self.every_n_epochs = int(every_n_epochs)

        self.prob_thr = float(prob_thr)
        self.nms_iou_thr = float(nms_iou_thr)
        self.min_area = int(min_area)
        self.max_candidates = int(max_candidates)
        self.vote_thr = float(vote_thr)
        self.iou_thr = float(iou_thr)
        self.sample_index = int(sample_index)

        self.local_max_footprint = int(local_max_footprint)
        self.explode_factor = float(explode_factor)

        self.early_vote_thr = float(early_vote_thr)
        self.early_max_candidates = int(early_max_candidates)

        self.ds = PannukePreparedDataset(
            root=self.data_root,
            split=self.split,
            transform=None,
            return_image_name=True,
            load_dists=True,
            n_rays=N_RAYS,
            return_dists=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        if epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        device = pl_module.device

        # ----- sample -----
        img_t, mask_t, types_t, dist_t, name = self.ds[self.sample_index]
        img_b = img_t.unsqueeze(0).to(device)

        gt_inst = mask_t.detach().cpu().numpy().astype(np.int32)
        gt_dist = dist_t.detach().cpu().numpy().astype(np.float32)
        gt_n = int(gt_inst.max())
        fg = (gt_inst > 0)

        # ----- forward -----
        with torch.no_grad():
            prob_logits, dist_pos, class_logits = pl_module.model(img_b)

        prob_map = torch.sigmoid(prob_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        dist_map = dist_pos[0].detach().cpu().numpy().astype(np.float32)
        cls_prob = torch.softmax(class_logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)

        # ----- prob diagnostics -----
        flat, flat_stats = is_prob_map_flat(prob_map, std_thr=1e-3, range_thr=1e-3)

        prob_peak = _prob_peak(prob_map, dist_map)
        prob_thr_final, peak_max = _auto_thr(prob_peak, self.prob_thr)

        vote_thr_use = self.early_vote_thr if epoch < 5 else self.vote_thr
        max_cand_use = min(self.early_max_candidates, self.max_candidates) if epoch < 5 else self.max_candidates

        # nb candidates (avant polygons)
        ys, xs = pick_candidates(
            prob_map=prob_peak,
            prob_thr=float(prob_thr_final),
            use_local_maxima=True,
            local_max_footprint=int(self.local_max_footprint),
            max_candidates=int(max_cand_use),
            plateau_mode="skimage",
            jitter_eps=1e-6,
            jitter_seed=0,
        )
        nb_candidates = int(len(ys))

        print(
            f"[SANITY] epoch={epoch} image={name} "
            f"gt={gt_n} flat={int(flat)} prob_std={flat_stats['std']:.4g} prob_rng={flat_stats['range']:.4g} "
            f"peak_max={peak_max:.3f} thr_final={prob_thr_final:.3f} "
            f"cand={nb_candidates} max_cand={max_cand_use} vote_thr={vote_thr_use:.2f}"
        )

        # ----- helper: eval -----
        def eval_pred(tag: str, pred_inst: np.ndarray):
            res = ioumatch.evaluate_image(
                pred_inst.astype(np.int32),
                gt_inst.astype(np.int32),
                threshold=float(self.iou_thr),
                method="greedy",
                inclusive=False,
                normalize=False,
            )
            print(
                f"[SANITY {tag}] TP={res['tp']} FP={res['fp']} FN={res['fn']} "
                f"F1={res['f1']:.3f} | pred={int(pred_inst.max())} gt={gt_n}"
            )
            return res

        # ----- A) HARD sanity: GTSeed + GTDist -----
        gt_seed = _gt_seedmap_from_gt_dist(gt_inst, gt_dist)
        pred_gtseed_gtdist, _ = stardist_decode_fn(
            prob_map=gt_seed,
            dist_map=gt_dist,
            class_prob=None,
            prob_thr=0.5,                 # seeds are 1.0
            nms_iou_thr=float(self.nms_iou_thr),
            use_local_maxima=False,        # inutile: déjà sparse
            local_max_footprint=int(self.local_max_footprint),
            max_candidates=100000,
            min_area=int(self.min_area),
            vote_thr=float(vote_thr_use),
        )
        sanity_n = int(pred_gtseed_gtdist.max())
        if sanity_n != gt_n:
            print(f"[SANITY HARD_FAIL] GTSeed+GTDist mismatch: got {sanity_n} vs gt {gt_n}")

        # ----- B) PredProb + GTDist (test prob head, même si dist head pas prête) -----
        pred_predprob_gtdist, _ = stardist_decode_fn(
            prob_map=prob_peak,
            dist_map=gt_dist,
            class_prob=None,
            prob_thr=float(prob_thr_final),
            nms_iou_thr=float(self.nms_iou_thr),
            use_local_maxima=True,
            local_max_footprint=int(self.local_max_footprint),
            max_candidates=int(max_cand_use),
            min_area=int(self.min_area),
            vote_thr=float(vote_thr_use),
        )
        res_predprob_gtdist = eval_pred("PredProb+GTDist", pred_predprob_gtdist)

        # ----- C) GTSeed + PredDist (test dist head sans dépendre de prob head) -----
        pred_gtseed_preddist, _ = stardist_decode_fn(
            prob_map=gt_seed,
            dist_map=dist_map,
            class_prob=None,
            prob_thr=0.5,
            nms_iou_thr=float(self.nms_iou_thr),
            use_local_maxima=False,
            local_max_footprint=int(self.local_max_footprint),
            max_candidates=100000,
            min_area=int(self.min_area),
            vote_thr=float(vote_thr_use),
        )
        res_gtseed_preddist = eval_pred("GTSeed+PredDist", pred_gtseed_preddist)

        # ----- D) REAL decode: PredProb + PredDist -----
        pred_inst, _ = stardist_decode_fn(
            prob_map=prob_peak,
            dist_map=dist_map,
            class_prob=cls_prob,
            prob_thr=float(prob_thr_final),
            nms_iou_thr=float(self.nms_iou_thr),
            use_local_maxima=True,
            local_max_footprint=int(self.local_max_footprint),
            max_candidates=int(max_cand_use),
            min_area=int(self.min_area),
            vote_thr=float(vote_thr_use),
        )
        res_real = eval_pred("REAL", pred_inst)

        pred_n = int(pred_inst.max())
        exploded = (gt_n > 0) and (pred_n > self.explode_factor * gt_n) and (not flat)
        if exploded:
            print(f"[SANITY WARNING] instance explosion: pred={pred_n} vs gt={gt_n} (factor>{self.explode_factor})")

        # dist sanity
        if fg.any():
            gt_mean = float(gt_dist[:, fg].mean())
            pr_mean = float(dist_map[:, fg].mean())
            ratio = pr_mean / (gt_mean + 1e-6)
        else:
            gt_mean = pr_mean = ratio = 0.0

        # log to wandb
        if trainer.logger is not None:
            trainer.logger.experiment.log({
                "sanity/epoch": epoch,
                "sanity/gt_n": gt_n,
                "sanity/pred_n": pred_n,
                "sanity/sanity_gtseed_gtdist_n": sanity_n,
                "sanity/prob_flat": int(flat),
                "sanity/prob_std": flat_stats["std"],
                "sanity/prob_range": flat_stats["range"],
                "sanity/peak_max": peak_max,
                "sanity/prob_thr_final": prob_thr_final,
                "sanity/nb_candidates": nb_candidates,
                "sanity/vote_thr_use": vote_thr_use,
                "sanity/max_candidates_use": max_cand_use,
                "sanity/exploded": int(exploded),
                "sanity/dist_mean_fg": pr_mean,
                "sanity/gt_dist_mean_fg": gt_mean,
                "sanity/dist_ratio_mean": ratio,

                "sanity/f1_predprob_gtdist": float(res_predprob_gtdist["f1"]),
                "sanity/f1_gtseed_preddist": float(res_gtseed_preddist["f1"]),
                "sanity/f1_real": float(res_real["f1"]),
            })

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
        n_nucleus_classes=N_NUCLEUS_CLASSES,
        w_prob=W_PROB,
        w_dist=W_DIST,
        w_ce=W_CE,
        w_tversky=W_TVERSKY,
        tversky_alpha=TV_ALPHA,
        tversky_beta=TV_BETA,
        tversky_include_background=TV_INCLUDE_BG,
        normalize_dist=NORMALIZE_DIST,
        normalize_div=NORMALIZE_DIV,
    )

    # wandb_logger = None
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        log_model=True,
    )


    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_callback = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="stardist_pannuke_best",
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
        vote_thr=VOTE_THR,
        iou_thr=SANITY_IOU_THR,
        sample_index=SANITY_IMAGE_INDEX,
        local_max_footprint=LOCAL_MAX_FOOTPRINT,
        explode_factor=10.0,  # baisser à 5 si on veux strict
        early_vote_thr=0.8,
        early_max_candidates=200,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=1,
        logger=wandb_logger,
        callbacks=[ckpt_callback, early_stop, TQDMProgressBar(refresh_rate=1), sanity_cb],
        log_every_n_steps=10,
        enable_progress_bar=True,
        accumulate_grad_batches=4,
        precision="16-mixed",
    )

    trainer.fit(model, datamodule=dm, ckpt_path=None)

    if dm.test_dataloader() is not None:
        trainer.test(model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    main()
