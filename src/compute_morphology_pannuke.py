from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops


def compute_pannuke_morphology(
    prepared_root: str,
    out_csv: str | None = None,
):
    """
    Parcourt les masques d'instances PanNuke préparés et
    calcule les descripteurs de forme pour chaque cellule.

    - prepared_root : dossier contenant train/val/test avec:
        images/img_XXXXXX.npy
        masks/mask_XXXXXX.npy

    - out_csv : chemin du CSV de sortie
    """
    root = Path(prepared_root)
    if out_csv is None:
        out_csv = root / "pannuke_morphology_features.csv"
    else:
        out_csv = Path(out_csv)

    rows = []

    for split in ["train", "val", "test"]:
        img_dir = root / split / "images"
        mask_dir = root / split / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            print(f"[WARN] Split {split} introuvable, on saute.")
            continue

        img_files = sorted(img_dir.glob("img_*.npy"))

        for img_path in tqdm(img_files, desc=f"Features {split}"):
            mask_path = mask_dir / img_path.name.replace("img_", "mask_")

            if not mask_path.exists():
                print(f"[WARN] Pas de masque pour {img_path.name}, on saute.")
                continue

            mask = np.load(mask_path)  # (H, W), uint16

            # regionprops ignore automatiquement le label 0 (fond)
            props = regionprops(mask)

            for p in props:
                area = float(p.area)
                perimeter = float(p.perimeter or 0.0)
                ecc = float(p.eccentricity)
                solidity = float(p.solidity)

                if perimeter > 0:
                    circularity = 4.0 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = np.nan

                rows.append(
                    {
                        "split": split,
                        "image": img_path.name,
                        "label": int(p.label),
                        "area": area,
                        "perimeter": perimeter,
                        "eccentricity": ecc,
                        "solidity": solidity,
                        "circularity": circularity,
                        "centroid_row": float(p.centroid[0]),
                        "centroid_col": float(p.centroid[1]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Morphology features saved to {out_csv} ({len(df)} cellules)")


if __name__ == "__main__":
    base_dir = Path(
        "/run/user/1000/gvfs/smb-share:server=zeus.pasteur.fr,share=bia/ayehadji/projet0"
    )
    prepared_root = base_dir / "data" / "prepared" / "pannuke"

    out_csv = prepared_root / "pannuke_morphology_features.csv"

    print(f"Prepared root : {prepared_root}")
    print(f"Out CSV       : {out_csv}")

    compute_pannuke_morphology(
        prepared_root=str(prepared_root),
        out_csv=str(out_csv),
    )
