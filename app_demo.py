# app_demo.py
"""
Application Gradio de démonstration pour la segmentation cellulaire.

Permet de comparer visuellement U-Net, StarDist et Cellpose sur des images.

Usage:
    python app_demo.py

L'interface sera accessible sur http://localhost:7860
"""

from pathlib import Path
import numpy as np
import torch
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, h_maxima

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data" / "prepared" / "pannuke"
CKPT_UNET = SCRIPT_DIR / "models" / "checkpoints" / "unet_pannuke_lit_best.ckpt"
CKPT_STARDIST = SCRIPT_DIR / "models" / "checkpoints" / "stardist_pannuke_best.ckpt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --------------------------------------------------------------------
# Chargement des modèles (lazy loading)
# --------------------------------------------------------------------

_MODELS = {}

def get_unet():
    if "unet" not in _MODELS:
        from models.lit_unet_pannuke import UNetLightning
        _MODELS["unet"] = UNetLightning.load_from_checkpoint(
            str(CKPT_UNET), map_location=DEVICE
        ).to(DEVICE).eval()
        print("U-Net loaded")
    return _MODELS["unet"]

def get_stardist():
    if "stardist" not in _MODELS:
        from models.lit_stardist_pannuke import StarDistLightning
        _MODELS["stardist"] = StarDistLightning.load_from_checkpoint(
            str(CKPT_STARDIST), map_location=DEVICE
        ).to(DEVICE).eval()
        print("StarDist loaded")
    return _MODELS["stardist"]

def get_cellpose():
    if "cellpose" not in _MODELS:
        from cellpose import models as cp_models
        _MODELS["cellpose"] = cp_models.CellposeModel(gpu=torch.cuda.is_available())
        print("Cellpose loaded")
    return _MODELS["cellpose"]

# --------------------------------------------------------------------
# Fonctions de prédiction
# --------------------------------------------------------------------

def unet_prob_to_instances(prob, thr=0.5, min_size=20, h=1.5):
    """Post-processing U-Net: watershed sur distance transform."""
    bin_mask = prob > thr
    bin_mask = remove_small_objects(bin_mask, min_size=min_size)
    if bin_mask.sum() == 0:
        return np.zeros_like(prob, dtype=np.int32)
    dist = ndi.distance_transform_edt(bin_mask)
    seeds = h_maxima(dist, h=h)
    seeds &= bin_mask
    markers, _ = ndi.label(seeds)
    if markers.max() == 0:
        markers, _ = ndi.label(bin_mask)
    labels = watershed(-dist, markers, mask=bin_mask)
    return labels.astype(np.int32)


def _prob_peak(prob_map: np.ndarray, dist_map: np.ndarray) -> np.ndarray:
    """Transforme prob_map en prob_peak pour StarDist."""
    center = dist_map.mean(axis=0)
    center = center / (float(center.max()) + 1e-6)
    return prob_map * center


def _auto_thr(prob_in: np.ndarray, base_thr: float):
    """Ajuste dynamiquement le seuil."""
    peak_max = float(prob_in.max())
    thr = min(float(base_thr), 0.9 * peak_max)
    thr = max(thr, 0.05)
    return thr, peak_max


@torch.no_grad()
def predict_unet(img_chw: np.ndarray):
    """Prédiction U-Net."""
    model = get_unet()
    x = torch.from_numpy(img_chw).unsqueeze(0).float().to(DEVICE)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
    inst = unet_prob_to_instances(prob, thr=0.5, min_size=20, h=1.5)
    return inst, prob


@torch.inference_mode()
def predict_stardist(img_chw: np.ndarray):
    """Prédiction StarDist avec correction prob_peak."""
    from models.stardist import stardist_decode

    model = get_stardist()
    x = torch.from_numpy(img_chw).unsqueeze(0).float().to(DEVICE)

    prob_logits, dist_pos, class_logits = model(x)
    prob_map = torch.sigmoid(prob_logits)[0, 0].cpu().numpy().astype(np.float32)
    dist_map = dist_pos[0].cpu().numpy().astype(np.float32)
    class_prob = torch.softmax(class_logits, dim=1)[0].cpu().numpy().astype(np.float32)

    # Correction prob_peak
    prob_peak = _prob_peak(prob_map, dist_map)
    prob_thr_final, _ = _auto_thr(prob_peak, 0.35)

    dec = stardist_decode(
        prob_map=prob_peak,
        dist_map=dist_map,
        class_prob=class_prob,
        prob_thr=float(prob_thr_final),
        nms_iou_thr=0.3,
        min_area=10,
        max_candidates=500,
        vote_thr=0.5,
        use_local_maxima=True,
        local_max_footprint=11,
    )

    pred_inst = dec[0] if isinstance(dec, tuple) else dec
    return pred_inst.astype(np.int32), prob_map


def predict_cellpose(img_chw: np.ndarray):
    """Prédiction Cellpose."""
    model = get_cellpose()
    img = np.transpose(img_chw, (1, 2, 0))  # HWC
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    masks, flows, styles = model.eval(img, diameter=None, do_3D=False)
    return masks.astype(np.int32), None


# --------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------

def create_colormap(n_instances: int):
    """Crée une colormap pour les instances."""
    if n_instances == 0:
        return np.zeros((1, 4))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    if n_instances > 20:
        colors = np.vstack([colors] * ((n_instances // 20) + 1))
    cmap = np.zeros((n_instances + 1, 4))
    cmap[0] = [0, 0, 0, 0]  # Background transparent
    cmap[1:n_instances+1] = colors[:n_instances]
    cmap[1:, 3] = 0.7  # Semi-transparent
    return cmap


def overlay_instances(image: np.ndarray, instances: np.ndarray, alpha: float = 0.5):
    """Superpose les instances colorées sur l'image."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    n_inst = int(instances.max())
    cmap = create_colormap(n_inst)

    # Créer l'overlay
    overlay = np.zeros((*instances.shape, 4), dtype=np.float32)
    for i in range(n_inst + 1):
        mask = instances == i
        overlay[mask] = cmap[i]

    # Blend
    overlay_rgb = (overlay[:, :, :3] * 255).astype(np.uint8)
    overlay_alpha = overlay[:, :, 3:4]

    result = image.astype(np.float32) * (1 - overlay_alpha * alpha) + \
             overlay_rgb.astype(np.float32) * overlay_alpha * alpha

    return result.astype(np.uint8)


def create_contour_overlay(image: np.ndarray, instances: np.ndarray):
    """Crée une visualisation avec contours des instances."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    result = image.copy()

    # Trouver les contours
    for inst_id in range(1, int(instances.max()) + 1):
        mask = (instances == inst_id).astype(np.uint8)
        # Dilate - erode = contour
        dilated = ndi.binary_dilation(mask, iterations=1)
        contour = dilated & ~mask

        # Couleur aléatoire mais reproductible
        np.random.seed(inst_id)
        color = np.random.randint(100, 255, 3)
        result[contour] = color

    return result


# --------------------------------------------------------------------
# Fonction principale de prédiction
# --------------------------------------------------------------------

def process_image(
    input_image,
    model_choice: str,
    visualization: str,
):
    """
    Traite une image avec le modèle sélectionné.

    Args:
        input_image: Image PIL ou numpy array
        model_choice: "U-Net", "StarDist", "Cellpose", ou "Tous les modèles"
        visualization: "Overlay coloré" ou "Contours"

    Returns:
        Tuple d'images selon le choix
    """
    if input_image is None:
        return None, "Veuillez charger une image."

    # Convertir en numpy
    if isinstance(input_image, Image.Image):
        img = np.array(input_image)
    else:
        img = input_image

    # Normaliser si nécessaire
    if img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)

    # S'assurer que c'est RGB
    if img_float.ndim == 2:
        img_float = np.stack([img_float] * 3, axis=-1)

    # Resize à 256x256 si nécessaire
    original_size = img_float.shape[:2]
    if img_float.shape[0] != 256 or img_float.shape[1] != 256:
        from skimage.transform import resize
        img_float = resize(img_float, (256, 256), preserve_range=True).astype(np.float32)

    # Format CHW pour le modèle
    img_chw = np.transpose(img_float, (2, 0, 1))

    results = {}
    stats = []

    models_to_run = []
    if model_choice == "Tous les modèles":
        models_to_run = ["U-Net", "StarDist", "Cellpose"]
    else:
        models_to_run = [model_choice]

    for model_name in models_to_run:
        try:
            if model_name == "U-Net":
                pred_inst, prob = predict_unet(img_chw)
            elif model_name == "StarDist":
                pred_inst, prob = predict_stardist(img_chw)
            elif model_name == "Cellpose":
                pred_inst, prob = predict_cellpose(img_chw)
            else:
                continue

            n_instances = int(pred_inst.max())
            stats.append(f"**{model_name}**: {n_instances} cellules détectées")

            # Visualisation
            if visualization == "Overlay coloré":
                vis = overlay_instances(img_float, pred_inst, alpha=0.6)
            else:
                vis = create_contour_overlay(img_float, pred_inst)

            results[model_name] = vis

        except Exception as e:
            stats.append(f"**{model_name}**: Erreur - {str(e)}")
            results[model_name] = img

    stats_text = "\n".join(stats)

    # Retourner selon le nombre de modèles
    if len(models_to_run) == 1:
        return results.get(models_to_run[0], img), stats_text
    else:
        # Créer une mosaïque
        imgs = [results.get(m, img) for m in ["U-Net", "StarDist", "Cellpose"]]
        # Ajouter labels
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, name in zip(axes, imgs, ["U-Net", "StarDist", "Cellpose"]):
            ax.imshow(im)
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.axis('off')
        plt.tight_layout()

        # Sauvegarder en image
        fig.canvas.draw()
        mosaic = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        mosaic = mosaic.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return mosaic, stats_text


def load_sample_image(sample_name: str):
    """Charge une image d'exemple du dataset."""
    if sample_name == "Aucun":
        return None

    try:
        # Extraire l'index de l'image
        idx = int(sample_name.split("_")[1].split(".")[0])
        split = "val"

        img_path = DATA_ROOT / split / "images" / f"img_{idx:06d}.npy"
        if img_path.exists():
            img = np.load(img_path)
            return (img * 255).astype(np.uint8)
    except Exception as e:
        print(f"Erreur chargement sample: {e}")

    return None


# --------------------------------------------------------------------
# Interface Gradio
# --------------------------------------------------------------------

def create_demo():
    """Crée l'interface Gradio."""

    # Charger la liste des images d'exemple
    sample_images = ["Aucun"]
    if DATA_ROOT.exists():
        val_dir = DATA_ROOT / "val" / "images"
        if val_dir.exists():
            files = sorted(val_dir.glob("img_*.npy"))[:20]  # 20 premiers
            sample_images += [f.stem for f in files]

    with gr.Blocks(
        title="Segmentation Cellulaire - Démo",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # Segmentation d'Instances Cellulaires

        **Stage M2 - Institut Pasteur**

        Cette application compare trois approches de segmentation cellulaire:
        - **U-Net**: Segmentation sémantique + watershed
        - **StarDist**: Détection de polygones star-convexes
        - **Cellpose**: Champs de gradients pour la séparation

        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Entrée")

                input_image = gr.Image(
                    label="Image à segmenter",
                    type="numpy",
                    height=300,
                )

                sample_dropdown = gr.Dropdown(
                    choices=sample_images,
                    value="Aucun",
                    label="Ou charger un exemple PanNuke",
                )

                model_choice = gr.Radio(
                    choices=["U-Net", "StarDist", "Cellpose", "Tous les modèles"],
                    value="Tous les modèles",
                    label="Modèle(s)",
                )

                viz_choice = gr.Radio(
                    choices=["Overlay coloré", "Contours"],
                    value="Overlay coloré",
                    label="Visualisation",
                )

                run_btn = gr.Button("Segmenter", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Résultat")

                output_image = gr.Image(
                    label="Segmentation",
                    height=400,
                )

                stats_output = gr.Markdown(
                    label="Statistiques",
                    value="*Les statistiques apparaîtront ici après la segmentation.*"
                )

        gr.Markdown("""
        ---

        ### À propos des modèles

        | Modèle | Approche | Forces | Faiblesses |
        |--------|----------|--------|------------|
        | **U-Net** | Segmentation sémantique + watershed | Bon recall, robuste | Difficultés de séparation |
        | **StarDist** | Polygones star-convexes | Rapide, bon pour cellules rondes | Sensible aux cellules irrégulières |
        | **Cellpose** | Champs de gradients | Meilleure séparation | Plus lent |

        ---
        *Projet réalisé dans le cadre du stage M2 à l'Institut Pasteur*
        """)

        # Events
        sample_dropdown.change(
            fn=load_sample_image,
            inputs=[sample_dropdown],
            outputs=[input_image],
        )

        run_btn.click(
            fn=process_image,
            inputs=[input_image, model_choice, viz_choice],
            outputs=[output_image, stats_output],
        )

        # Permettre aussi de lancer avec Enter sur l'image
        input_image.change(
            fn=lambda: None,  # Reset
            inputs=[],
            outputs=[],
        )

    return demo


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Démarrage de l'application de démonstration")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    print(f"U-Net checkpoint: {CKPT_UNET.exists()}")
    print(f"StarDist checkpoint: {CKPT_STARDIST.exists()}")
    print("=" * 60)

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Accessible depuis le réseau local
        server_port=7860,
        share=False,  # Mettre True pour un lien public
        show_error=True,
    )
