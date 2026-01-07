import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs
from torch.utils.data import DataLoader
import argparse
import json
import cv2  # Per salvare le immagini

# --- Importa i moduli del tuo progetto ---
from federated_semantic_segmentation.task import Net
from federated_semantic_segmentation.utils.metrics import Metrics
from federated_semantic_segmentation.utils.data import denorm_rgb
from federated_semantic_segmentation.utils.uavid_data import UAVidDataset

# ---------- CONFIG DI DEFAULT (MODIFICA PURE) ----------

DEFAULT_MODEL_PATH = "checkpoints/iid_100ls_20tr_30cl_last_1.pt"
DEFAULT_OUTPUT_DIR = "uavid_eval_non_iid_20tr_30cl"
DEFAULT_IMG_DIR    = "C://Users/Lab/Desktop/UAVid/clean/rgb"
DEFAULT_LABEL_DIR  = "C://Users/Lab/Desktop/UAVid/clean/label"

# Quante immagini salvare per ispezione qualitativa
NUM_IMAGES_TO_LOG = 20

# Info esperimento da salvare nel report
HYPERPARAMS = {
    "model_path": DEFAULT_MODEL_PATH,
    "dataset": "UAVid (real images)",
    "subset": "all",          # clean / shifted / flipped / all
    "lr": 0.0005,
    "local_steps": 100,
    "total_rounds": 20,
    "num_clients": 30,
    "splits": "non_iid",
    "seed": 1,
}

# ---------- MAPPATURA 28 -> 8 CLASSI COARSE ----------

# Indici delle classi SynDrone (28 fine) -> coarse:
# 0: Road, 1: Nature, 2: Person, 3: Vehicle,
# 4: Construction, 5: Obstacle, 6: Water, 7: Void

# label_names del tuo dataset SynDrone (ordine delle 28 classi):
# 0  Building
# 1  Fence
# 2  Other
# 3  Pole
# 4  RoadLine
# 5  Road
# 6  Sidewalk
# 7  Vegetation
# 8  Wall
# 9  Traffic Signs
# 10 Sky
# 11 Ground
# 12 Bridge
# 13 Rail Track
# 14 Guard Rail
# 15 Traffic Light
# 16 Static
# 17 Dynamic
# 18 Water
# 19 Terrain
# 20 Person
# 21 Rider
# 22 Car
# 23 Truck
# 24 Bus
# 25 Train
# 26 Motorcycle
# 27 Bicycle

FINE_TO_COARSE_LIST = [
    4,  # 0  Building      -> Construction
    4,  # 1  Fence         -> Construction
    5,  # 2  Other         -> Obstacle
    5,  # 3  Pole          -> Obstacle
    0,  # 4  RoadLine      -> Road
    0,  # 5  Road          -> Road
    0,  # 6  Sidewalk      -> Road
    1,  # 7  Vegetation    -> Nature
    4,  # 8  Wall          -> Construction
    5,  # 9  Traffic Signs -> Obstacle
    7,  # 10 Sky           -> Void
    0,  # 11 Ground        -> Road
    4,  # 12 Bridge        -> Construction
    0,  # 13 Rail Track    -> Road
    5,  # 14 Guard Rail    -> Obstacle
    5,  # 15 Traffic Light -> Obstacle
    5,  # 16 Static        -> Obstacle
    5,  # 17 Dynamic       -> Obstacle
    6,  # 18 Water         -> Water
    1,  # 19 Terrain       -> Nature
    2,  # 20 Person        -> Person
    2,  # 21 Rider         -> Person
    3,  # 22 Car           -> Vehicle
    3,  # 23 Truck         -> Vehicle
    3,  # 24 Bus           -> Vehicle
    3,  # 25 Train         -> Vehicle
    3,  # 26 Motorcycle    -> Vehicle
    3,  # 27 Bicycle       -> Vehicle
]

FINE_TO_COARSE_TORCH = torch.tensor(FINE_TO_COARSE_LIST, dtype=torch.long)


def run_offline_analysis(
    model_path: str,
    output_dir: str,
    img_dir: str,
    label_dir: str,
    subset_name: str = "all",
):
    """
    Esegue analisi offline su UAVid:
      1. Valuta il modello sul test set (mIoU sulle 8 classi coarse).
      2. Salva immagini RGB/GT/Pred.
      3. Salva un 'report.txt' con iperparametri e tabella metriche.
    """

    print(f"--- Avvio Analisi Offline UAVid ---")
    print(f"Modello: {model_path}")
    print(f"Immagini: {img_dir}")
    print(f"Label   : {label_dir}")
    print(f"Output  : {output_dir}")

    # 1. Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device utilizzato: {device}")

    # Crea le cartelle di output
    output_rgb_dir  = path.join(output_dir, "rgb_images")
    output_gt_dir   = path.join(output_dir, "gt_labels")
    output_pred_dir = path.join(output_dir, "pred_predictions")
    makedirs(output_rgb_dir, exist_ok=True)
    makedirs(output_gt_dir, exist_ok=True)
    makedirs(output_pred_dir, exist_ok=True)

    # 2. Carica il Modello
    try:
        model = Net()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Modello caricato con successo.")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il modello da {model_path}. {e}")
        return

    # 3. Costruisci Dataset e DataLoader UAVid
    try:
        dataset = UAVidDataset(
            img_dir=img_dir,
            label_dir=label_dir,
            subset_name=subset_name,
            fake_altitude_m=50.0,
        )
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        cnames = dataset.coarse_labels
        color_label_fn = dataset.color_label_coarse

        print(f"Dati di test UAVid: {len(dataset)} immagini.")
        print("Classi coarse:", cnames)
    except Exception as e:
        print(f"ERRORE: Impossibile costruire Dataset/DataLoader UAVid. {e}")
        return

    # 4. Inizializza Metrics per le 8 classi coarse
    metrics_calculator = Metrics(cnames, log_colors=False, device=device)

    # 5. Loop di Valutazione
    print("Esecuzione valutazione in corso...")
    with torch.no_grad():
        for i, (x4, mlb_coarse, alt, meta) in enumerate(val_loader):

            x4  = x4.to(device, dtype=torch.float32)   # [1, 4, H, W]
            mlb_coarse = mlb_coarse.to(device, dtype=torch.long)  # [1, H, W]
            alt = alt.to(device, dtype=torch.float32)  # [1]

            # Forward del modello (output 28 classi fine)
            logits = model(x4, alt)       # [1, 28, H, W]
            preds_fine = logits.argmax(dim=1)  # [1, H, W] in [0..27]

            # Mappa 28 -> 8 classi coarse tramite lookup tensor
            ftc = FINE_TO_COARSE_TORCH.to(device)
            preds_coarse = ftc[preds_fine]      # [1, H, W] in [0..7]

            # Aggiorna metriche globali (su coarse space)
            metrics_calculator.add_sample(preds_coarse, mlb_coarse)

            # Salva solo i primi N campioni di immagini
            if i < NUM_IMAGES_TO_LOG:
                try:
                    rgb_tensor = x4[0, :3, :, :]         # [3,H,W]
                    gt_mask    = mlb_coarse[0]           # [H,W]
                    pred_mask  = preds_coarse[0]         # [H,W]

                    # RGB "denormalizzato"
                    rgb_vis = denorm_rgb(
                        rgb_tensor.permute(1, 2, 0).cpu().numpy()
                    )
                    gt_vis = color_label_fn(gt_mask.cpu().numpy())
                    pred_vis = color_label_fn(pred_mask.cpu().numpy())

                    sample_name = f"{meta['filename'][0]}_uavid.png"

                    rgb_save  = (rgb_vis * 255.0).astype(np.uint8)
                    rgb_save  = cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR)
                    gt_save   = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
                    pred_save = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(path.join(output_rgb_dir,  sample_name), rgb_save)
                    cv2.imwrite(path.join(output_gt_dir,   sample_name), gt_save)
                    cv2.imwrite(path.join(output_pred_dir, sample_name), pred_save)

                except Exception as e:
                    print(f"Errore durante il salvataggio dell'immagine {i}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  ...elaborati {i+1}/{len(val_loader)} batch")

    print("Valutazione completata. Calcolo e salvataggio report...")

    # 6. Salva il report.txt con iperparametri + metriche
    report_file_path = path.join(output_dir, "report.txt")
    HYPERPARAMS["model_path"] = model_path
    HYPERPARAMS["dataset_img_dir"] = img_dir
    HYPERPARAMS["dataset_label_dir"] = label_dir
    HYPERPARAMS["subset"] = subset_name

    with open(report_file_path, "w") as f:
        # A. Iperparametri
        f.write("=" * 70 + "\n")
        f.write(" EXPERIMENT HYPERPARAMETERS\n")
        f.write("=" * 70 + "\n")
        f.write(json.dumps(HYPERPARAMS, indent=4))
        f.write("\n\n")

        # B. Tabella delle metriche
        f.write("=" * 70 + "\n")
        f.write(" DETAILED METRICS PER CLASS (UAVid, coarse 8 classes)\n")
        f.write("=" * 70 + "\n")
        metrics_table = str(metrics_calculator)
        f.write(metrics_table)
        f.write("\n\n")

        # C. mIoU finale
        f.write("=" * 70 + "\n")
        f.write(" FINAL mIoU (UAVid, coarse)\n")
        f.write("=" * 70 + "\n")
        global_miou = metrics_calculator.percent_mIoU()
        f.write(f"mIoU: {global_miou:.4f}%\n")

    print(f"--- Analisi Offline UAVid Completata ---")
    print(f"Report e immagini salvate in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline evaluation su UAVid (real images)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Percorso al modello .pt salvato.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella dove salvare report e immagini.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=DEFAULT_IMG_DIR,
        help="Cartella con le immagini RGB di UAVid.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=DEFAULT_LABEL_DIR,
        help="Cartella con le label (palette RGB) di UAVid.",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="all",
        help="Nome del subset (clean / shifted / flipped / all) solo per logging.",
    )
    args = parser.parse_args()

    run_offline_analysis(
        model_path=args.model_path,
        output_dir=args.output_dir,
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        subset_name=args.subset_name,
    )
