from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path("/")
csv_path = base_dir / "data" / "prepared" / "pannuke" / "pannuke_morphology_features.csv"

df = pd.read_csv(csv_path)

# Filtrer les NaN de circularity si besoin
df["circularity"] = df["circularity"].fillna(0.0)

def assign_morph(row):
    circ = row["circularity"]
    ecc = row["eccentricity"]
    sol = row["solidity"]
    area = row["area"]

    # 1) Rondes : convexes, peu allongées
    if (circ > 0.80) and (ecc < 0.65) and (sol > 0.9):
        return "round"

    # 2) Allongées : bien elliptiques, assez pleines
    if (ecc > 0.80) and (sol > 0.85):
        return "elongated"

    # 3) Irrégulières : bords bizarres, pas très convexes,
    # ou très grosses (souvent artefacts / structures complexes)
    if (circ < 0.55) or (sol < 0.80) or (area > df["area"].quantile(0.99)):
        return "irregular"

    # 4) Sinon : morphologie "intermédiaire"
    return "other"

df["morph_class"] = df.apply(assign_morph, axis=1)

out_csv = csv_path.with_name("pannuke_morphology_with_classes.csv")
df.to_csv(out_csv, index=False)
print("Saved with morph classes ->", out_csv)
print(df["morph_class"].value_counts())
