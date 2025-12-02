from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.ndimage import label as nd_label


def prepare_pannuke_instances(
    pannuke_root: str = "data/pannuke",
    out_root: str = "data/prepared/pannuke",
):
    """
    Prépare PanNuke au format canonique :
    - images float32 dans [0, 1], shape (H, W, 3)
    - masques d'instances uint16, shape (H, W), 0 = fond, >0 = id cellule
    - splités en train/val/test selon fold1/2/3
    """
    pannuke_root = Path(pannuke_root)
    out_root = Path(out_root)

    if not pannuke_root.exists():
        print(f"Erreur : dossier source {pannuke_root} introuvable.")
        return

    fold_to_split = {
        "fold1": "train",
        "fold2": "val",
        "fold3": "test",
    }

    # Création des dossiers de sortie
    for split in fold_to_split.values():
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)

    global_idx = 0

    for fold_name, split in fold_to_split.items():
        fold_dir = pannuke_root / fold_name
        images_npy = fold_dir / "images.npy"
        masks_npy = fold_dir / "masks.npy"

        if not images_npy.exists() or not masks_npy.exists():
            print(f"Fichiers manquants dans {fold_name}, on saute...")
            continue

        images = np.load(images_npy)  # (N, H, W, 3)
        masks = np.load(masks_npy)    # (N, H, W, C)

        split_img_dir = out_root / split / "images"
        split_mask_dir = out_root / split / "masks"

        for i in tqdm(range(images.shape[0]), desc=f"{fold_name} -> {split}"):
            img = images[i]
            mask_raw = masks[i]

            # On saute les images sans cellules
            if mask_raw.sum() == 0:
                continue

            # Normalisation en [0, 1]
            img = img.astype(np.float32) / 255.0

            # Construction du masque d'instances
            mask_instance = np.zeros(mask_raw.shape[:2], dtype=np.uint16)
            current_label = 1

            for c in range(mask_raw.shape[-1] - 1):  # on ignore le dernier canal (background)
                binary = mask_raw[..., c] > 0
                labeled, n = nd_label(binary)

                if n == 0:
                    continue

                labeled_nonzero = labeled > 0
                labeled[labeled_nonzero] += (current_label - 1)
                mask_instance[labeled_nonzero] = labeled[labeled_nonzero]
                current_label += n

            global_idx += 1

            np.save(split_img_dir / f"img_{global_idx:06d}.npy", img)
            np.save(split_mask_dir / f"mask_{global_idx:06d}.npy", mask_instance)

    print("Préparation PanNuke terminée.")


if __name__ == "__main__":
    base_dir = Path(
        "/run/user/1000/gvfs/smb-share:server=zeus.pasteur.fr,share=bia/ayehadji/projet0"
    )
    pannuke_root = base_dir / "data" / "pannuke"

    # Dossier de sortie pour les données préparées PanNuke
    out_root = base_dir / "data" / "prepared" / "pannuke"

    print(f"Pannuke root : {pannuke_root}")
    print(f"Output root  : {out_root}")

    prepare_pannuke_instances(
        pannuke_root=str(pannuke_root),
        out_root=str(out_root),
    )
