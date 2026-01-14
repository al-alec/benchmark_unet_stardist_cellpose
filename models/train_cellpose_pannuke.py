# train_cellpose_pannuke.py
"""
Fine-tuning de Cellpose sur le dataset PanNuke.

Cellpose utilise son propre système d'entraînement (pas PyTorch Lightning).
Ce script prépare les données et lance le fine-tuning via l'API native.

Usage:
    python models/train_cellpose_pannuke.py

Le modèle fine-tuné sera sauvegardé dans models/checkpoints/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from tqdm import tqdm
import os

# Cellpose imports
from cellpose import models, train, io

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

# Chemins absolus basés sur l'emplacement du script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = PROJECT_ROOT / "data" / "prepared" / "pannuke"
CKPT_DIR = SCRIPT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"CKPT_DIR: {CKPT_DIR}")

# Hyperparamètres de fine-tuning (Cellpose 4.x)
CONFIG = {
    "n_epochs": 100,
    "learning_rate": 1e-4,      # Cellpose 4.x default: 1e-5, on augmente un peu
    "weight_decay": 0.1,        # Cellpose default
    "batch_size": 4,            # Ajuster selon GPU RAM
    "min_train_masks": 1,       # Nombre minimum de masques par image
    "model_type": "cyto2",      # Modèle de base: 'cyto', 'cyto2', 'nuclei'
    "chan": 0,                  # 0 = grayscale, 1 = R, 2 = G, 3 = B
    "chan2": 0,                 # Second channel (0 = none)
    "rescale": False,           # Pas de rescale (images déjà 256x256)
    "normalize": True,          # Normalize images
    "save_every": 20,           # Sauvegarder tous les N epochs
}

# Nom du modèle fine-tuné
MODEL_NAME = "cellpose_pannuke_finetuned"


# --------------------------------------------------------------------
# Préparation des données
# --------------------------------------------------------------------

def load_pannuke_split(split: str, max_images: int | None = None):
    """
    Charge un split PanNuke au format Cellpose.

    Cellpose attend:
      - images: list of (H, W, C) arrays (uint8 ou float)
      - masks: list of (H, W) arrays (int, 0=bg, >0=instance)

    Returns:
        images: list[np.ndarray]
        masks: list[np.ndarray]
        names: list[str]
    """
    img_dir = DATA_ROOT / split / "images"
    mask_dir = DATA_ROOT / split / "masks"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing: {mask_dir}")

    img_files = sorted(img_dir.glob("img_*.npy"))
    if max_images is not None:
        img_files = img_files[:max_images]

    images = []
    masks = []
    names = []

    for img_path in tqdm(img_files, desc=f"Loading {split}"):
        name = img_path.name
        mask_path = mask_dir / name.replace("img_", "mask_")

        if not mask_path.exists():
            print(f"Warning: missing mask {mask_path}, skipping")
            continue

        # Charger image (H,W,3) float32 [0,1] -> convert to uint8
        img = np.load(img_path)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Charger masque (H,W) int
        mask = np.load(mask_path).astype(np.int32)

        images.append(img)
        masks.append(mask)
        names.append(name)

    print(f"Loaded {len(images)} images from {split}")
    return images, masks, names


def prepare_cellpose_data():
    """
    Prépare les données train/val pour Cellpose.
    """
    print("=" * 60)
    print("Préparation des données pour Cellpose fine-tuning")
    print("=" * 60)

    train_images, train_masks, train_names = load_pannuke_split("train")
    val_images, val_masks, val_names = load_pannuke_split("val")

    print(f"\nTrain: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")

    return {
        "train": (train_images, train_masks, train_names),
        "val": (val_images, val_masks, val_names),
    }


# --------------------------------------------------------------------
# Fine-tuning
# --------------------------------------------------------------------

def train_cellpose(
    train_images: list,
    train_masks: list,
    val_images: list | None = None,
    val_masks: list | None = None,
    model_type: str = "cyto2",
    n_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    batch_size: int = 4,
    min_train_masks: int = 1,
    save_path: str | None = None,
    save_every: int = 20,
    chan: int = 0,
    chan2: int = 0,
    rescale: bool = False,
    normalize: bool = True,
):
    """
    Fine-tune un modèle Cellpose sur des données custom (API Cellpose 4.x).

    Args:
        train_images: Liste d'images (H,W,C) uint8
        train_masks: Liste de masques (H,W) int
        val_images: Images de validation (optionnel)
        val_masks: Masques de validation (optionnel)
        model_type: 'cyto', 'cyto2', ou 'nuclei'
        n_epochs: Nombre d'epochs
        learning_rate: Learning rate initial
        weight_decay: Weight decay
        batch_size: Taille du batch
        min_train_masks: Minimum de masques par image
        save_path: Où sauvegarder le modèle
        save_every: Sauvegarder tous les N epochs
        chan: Channel principal (0=gray, 1=R, 2=G, 3=B)
        chan2: Second channel (0=none)
        rescale: Rescale images
        normalize: Normalize images

    Returns:
        model: Modèle fine-tuné
        model_path: Chemin du modèle sauvegardé
    """
    print("\n" + "=" * 60)
    print("Fine-tuning Cellpose (v4.x)")
    print("=" * 60)
    print(f"Model type: {model_type}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {batch_size}")
    print(f"Train images: {len(train_images)}")
    if val_images:
        print(f"Val images: {len(val_images)}")
    print("=" * 60 + "\n")

    # Initialiser le modèle de base
    model = models.CellposeModel(
        gpu=True,
        model_type=model_type,
    )

    # Déterminer le chemin de sauvegarde
    if save_path is None:
        save_path = str(CKPT_DIR)

    # Fine-tuning avec l'API native de Cellpose 4.x
    # Signature: train_seg(net, train_data, train_labels, ..., save_path, model_name, ...)
    model_path, train_losses, test_losses = train.train_seg(
        net=model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=val_images,
        test_labels=val_masks,
        normalize=normalize,
        rescale=rescale,
        save_path=save_path,
        save_every=save_every,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        min_train_masks=min_train_masks,
        model_name=MODEL_NAME,
    )

    print(f"\nModèle sauvegardé: {model_path}")
    print(f"Train loss finale: {train_losses[-1]:.4f}")
    if test_losses and len(test_losses) > 0:
        print(f"Val loss finale: {test_losses[-1]:.4f}")

    return model, model_path


# --------------------------------------------------------------------
# Évaluation
# --------------------------------------------------------------------

def evaluate_cellpose(
    model_path: str,
    test_images: list,
    test_masks: list,
    iou_threshold: float = 0.5,
):
    """
    Évalue un modèle Cellpose fine-tuné.
    """
    import ioumatch

    print("\n" + "=" * 60)
    print("Évaluation du modèle fine-tuné")
    print("=" * 60)

    # Charger le modèle fine-tuné
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=model_path,
    )

    results = []
    for i, (img, gt_mask) in enumerate(tqdm(zip(test_images, test_masks), total=len(test_images), desc="Eval")):
        # Prédiction
        pred_mask, flows, styles = model.eval(img, diameter=None, do_3D=False)

        # Matching
        res = ioumatch.evaluate_image(
            pred_mask.astype(np.int32),
            gt_mask.astype(np.int32),
            threshold=iou_threshold,
            method="greedy",
            inclusive=False,
            normalize=False,
        )

        results.append({
            "TP": int(res["tp"]),
            "FP": int(res["fp"]),
            "FN": int(res["fn"]),
            "F1": float(res["f1"]),
            "n_pred": int(pred_mask.max()),
            "n_gt": int(gt_mask.max()),
        })

    # Agrégation
    total_TP = sum(r["TP"] for r in results)
    total_FP = sum(r["FP"] for r in results)
    total_FN = sum(r["FN"] for r in results)

    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\nRésultats sur {len(test_images)} images:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP/FP/FN:  {total_TP}/{total_FP}/{total_FN}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    """
    Pipeline complet: préparation, fine-tuning, évaluation.
    """
    # 1. Préparer les données
    data = prepare_cellpose_data()
    train_images, train_masks, _ = data["train"]
    val_images, val_masks, _ = data["val"]

    # 2. Fine-tuning
    model, model_path = train_cellpose(
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        **CONFIG,
    )

    # 3. Évaluation sur val
    print("\n" + "=" * 60)
    print("Évaluation finale")
    print("=" * 60)

    # Évaluer le modèle fine-tuné
    print("\n--- Modèle FINE-TUNÉ ---")
    evaluate_cellpose(model_path, val_images, val_masks)

    # Comparer avec le modèle de base (optionnel)
    print("\n--- Modèle de BASE (cyto2) pour comparaison ---")
    base_model = models.CellposeModel(gpu=True, model_type="cyto2")

    results_base = []
    for img, gt_mask in tqdm(zip(val_images, val_masks), total=len(val_images), desc="Eval base"):
        pred_mask, _, _ = base_model.eval(img, diameter=None, do_3D=False)
        import ioumatch
        res = ioumatch.evaluate_image(
            pred_mask.astype(np.int32),
            gt_mask.astype(np.int32),
            threshold=0.5,
            method="greedy",
        )
        results_base.append(res)

    total_TP = sum(r["tp"] for r in results_base)
    total_FP = sum(r["fp"] for r in results_base)
    total_FN = sum(r["fn"] for r in results_base)
    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Base model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    print("\n" + "=" * 60)
    print("Fine-tuning terminé!")
    print(f"Modèle sauvegardé: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
