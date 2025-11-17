import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs
from torch.utils.data import DataLoader
import argparse
import json
import cv2  # Per salvare le immagini
import sys # Per salvare l'output di print

# --- Importa i moduli del tuo progetto ---
from federated_semantic_segmentation.task import Net, load_server_data
from federated_semantic_segmentation.utils.metrics import Metrics
from federated_semantic_segmentation.utils.data import denorm_rgb, SelmaDrones

# --- IMPOSTAZIONI DELL'ANALISI ---
DEFAULT_MODEL_PATH = "final_model.pt"
# Cartella di output principale per questo esperimento
DEFAULT_OUTPUT_DIR = "run_analysis_1" 
NUM_IMAGES_TO_LOG = 20

# Aggiorna questi valori per ogni esperimento!
HYPERPARAMS = {
    "model_path": DEFAULT_MODEL_PATH,
    "lr": 0.0005,
    "local_epochs": 5,
    "total_rounds": 5,
    "batch_size": 2,
    "train_id": 600
}
# ------------------------------------

def run_offline_analysis(model_path: str, output_dir: str):
    """
    Esegue un'analisi offline completa:
    1. Salva immagini 1:1 (non stretchate) in cartelle separate.
    2. Salva un 'report.txt' con iperparametri e tabella metriche.
    """
    
    print(f"--- Avvio Analisi Offline ---")
    print(f"Modello: {model_path}")
    print(f"Output salvato in: {output_dir}")

    # 1. Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Crea le cartelle di output
    output_rgb_dir = path.join(output_dir, "rgb_images")
    output_gt_dir = path.join(output_dir, "gt_labels")
    output_pred_dir = path.join(output_dir, "pred_predictions")
    makedirs(output_rgb_dir, exist_ok=True)
    makedirs(output_gt_dir, exist_ok=True)
    makedirs(output_pred_dir, exist_ok=True)

    # 2. Carica il Modello
    try:
        model = Net()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Modello caricato con successo.")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il modello da {model_path}. {e}")
        return

    # 3. Carica i Dati di Test
    try:
        val_loader = load_server_data()
        cnames = val_loader.dataset.label_names
        color_label_fn = val_loader.dataset.color_label
        print(f"Dati di test ({len(val_loader.dataset)} campioni) caricati.")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare i dati. {e}")
        return

    # 4. Inizializza la classe Metrics
    # !!! --- MODIFICA CHIAVE --- !!!
    # Disattiviamo i colori ANSI per la scrittura su file
    metrics_calculator = Metrics(cnames, log_colors=False, device=device)
    # !!! --- FINE MODIFICA --- !!!

    # 5. Esegui il Loop di Valutazione
    print("Esecuzione valutazione in corso...")
    with torch.no_grad():
        for i, ((x4, mlb), _meta) in enumerate(val_loader):
            
            x4 = x4.to(device, dtype=torch.float32)
            mlb = mlb.to(device, dtype=torch.long)

            logits = model(x4)
            preds = logits.argmax(dim=1)

            metrics_calculator.add_sample(preds, mlb)

            # Salva solo i primi N campioni di immagini
            if i < NUM_IMAGES_TO_LOG:
                try:
                    rgb_tensor = x4[0, :3, :, :]
                    gt_mask = mlb[0]
                    pred_mask = preds[0]

                    # Prepara array NumPy visualizzabili (in formato RGB)
                    rgb_vis = denorm_rgb(rgb_tensor.permute(1, 2, 0).cpu().numpy())
                    gt_vis = color_label_fn(gt_mask.cpu().numpy())
                    pred_vis = color_label_fn(pred_mask.cpu().numpy())

                    # --- Salva file .png separati (non stretchati) ---
                    sample_name = f"sample_{i:03d}.png"
                    
                    rgb_save = (rgb_vis * 255.0).astype(np.uint8)
                    rgb_save = cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR)
                    gt_save = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
                    pred_save = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)
                    
                    cv2.imwrite(path.join(output_rgb_dir, sample_name), rgb_save)
                    cv2.imwrite(path.join(output_gt_dir, sample_name), gt_save)
                    cv2.imwrite(path.join(output_pred_dir, sample_name), pred_save)

                except Exception as e:
                    print(f"Errore durante il salvataggio dell'immagine {i}: {e}")

            if (i+1) % 100 == 0:
                print(f"  ...elaborati {i+1}/{len(val_loader)} campioni")

    print("Valutazione completata. Calcolo e salvataggio report...")

    # --- 6. Salva il report.txt ---
    
    # Definisci il percorso del file di report
    report_file_path = path.join(output_dir, "report.txt")
    
    with open(report_file_path, "w") as f:
        # A. Scrivi gli Iperparametri
        f.write("=" * 70 + "\n")
        f.write(" EXPERIMENT HYPERPARAMETERS\n")
        f.write("=" * 70 + "\n")
        # Aggiorna gli iperparametri con il nome del modello
        HYPERPARAMS["model_path"] = model_path
        f.write(json.dumps(HYPERPARAMS, indent=4))
        f.write("\n\n")

        # B. Scrivi la Tabella delle Metriche
        f.write("=" * 70 + "\n")
        f.write(" DETAILED METRICS PER CLASS\n")
        f.write("=" * 70 + "\n")
        
        # Ora questa stringa sarà pulita (senza colori)
        metrics_table = str(metrics_calculator)
        f.write(metrics_table)
        f.write("\n\n")

        # C. Scrivi il mIoU finale (per vederlo subito)
        f.write("=" * 70 + "\n")
        f.write(" FINAL mIoU\n")
        f.write("=" * 70 + "\n")
        global_miou = metrics_calculator.percent_mIoU()
        f.write(f"mIoU: {global_miou:.4f}%\n")

    print(f"--- Analisi Offline Completata ---")
    print(f"Report e immagini salvate in: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script di Analisi e Visualizzazione Offline")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help="Percorso al modello .pt salvato."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella dove salvare il report.txt e le immagini (rgb/, gt/, pred/)."
    )
    args = parser.parse_args()
    
    HYPERPARAMS["model_path"] = args.model_path
    
    run_offline_analysis(args.model_path, args.output_dir)