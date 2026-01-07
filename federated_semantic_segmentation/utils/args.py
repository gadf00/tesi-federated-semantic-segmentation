# args.py
import argparse

def str2bool(s):
    return s.lower() in ["1", "t", "true", "y", "yes"]

def get_args():
    parser = argparse.ArgumentParser(description="Centralized training for Early-Fusion RGB+D segmentation (LR-ASPP MobileNetV3)")

    # === Dataset paths ===
    parser.add_argument("--root_path", type=str, default="E:/selmadrones/renders",
                        help="Percorso alla root del dataset (cartelle TownXX_Opt_120).")
    parser.add_argument("--splits_path", type=str, default="E:/selmadrones/splits",
                        help="Percorso ai file degli split (train.txt / val.txt).")

    # === Dataset options ===
    parser.add_argument("--town", type=str, default="all",
                        choices=['01', '02', '03', '04', '05', '06', '07', '10HD', 'all'],
                        help="Città da includere (default=all).")
    parser.add_argument("--height", type=str, default="all",
                        choices=['20', '50', '80', 'all'],
                        help="Altezze da includere (default=all).")

    # === Training parameters ===
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche di training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Dimensione del batch.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate iniziale.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay AdamW.")
    parser.add_argument("--dloader_workers", type=int, default=4, help="Numero di worker per DataLoader.")
    parser.add_argument("--iters_per_epoch", type=int, default=3000, help="Iterazioni per epoca (debug).")

    # === Logging / Checkpoint ===
    parser.add_argument("--logdir", type=str, default="logs", help="Directory per TensorBoard / log.")
    parser.add_argument("--evaldir", type=str, default="evals", help="Directory per salvataggio risultati.")
    parser.add_argument("--override_logs", type=str2bool, default=False, help="Sovrascrive log esistenti.")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Checkpoint da cui riprendere il training.")

    # === Debug flag ===
    parser.add_argument("--debug", type=str2bool, default=False, help="Attiva modalità di debug.")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
