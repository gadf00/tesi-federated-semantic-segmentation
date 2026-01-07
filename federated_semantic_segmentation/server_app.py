import math
import os
import torch
import random
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg


from federated_semantic_segmentation.task import Net, get_evaluate_fn

# Crea l'app server
app = ServerApp()

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Rendere cuDNN il più deterministico possibile
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# COSINE LEARNING RATE PER ROUND
# ============================================================

def cosine_lr_round(
    round_idx: int,
    total_rounds: int,
    base_lr: float,
) -> float:
    """
    LR schedule dipendente dal numero totale di round:

    - total_rounds <= 10:      nessun decay, LR fisso = base_lr
    - total_rounds <= 20:      decay leggero nella seconda metà
    - total_rounds >= 40:      warmup metà training, poi cosine decay forte
    """

    # Caso 1: esperimenti piccoli (10 round)
    if total_rounds <= 10:
        return base_lr

    # Caso 2: esperimenti medi (20 round) -> decay leggero nella seconda metà
    if total_rounds <= 20:
        decay_start = total_rounds // 2    # es. 20 -> 10
        min_lr = base_lr * 0.3             # scende solo a ~30% di base_lr

        if round_idx <= decay_start:
            return base_lr

        # Cosine sui round [decay_start+1 .. total_rounds]
        effective_rounds = total_rounds - decay_start
        t = (round_idx - decay_start) / effective_rounds
        t = max(0.0, min(1.0, t))
        cos_term = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0

        return min_lr + (base_lr - min_lr) * cos_term

    # Caso 3: esperimenti lunghi (40, 80, ...) -> warmup metà, poi decay serio
    decay_start = total_rounds // 2        # es. 40->20, 80->40
    min_lr = 1e-6                          # scende molto

    if round_idx <= decay_start:
        return base_lr

    effective_rounds = total_rounds - decay_start
    t = (round_idx - decay_start) / effective_rounds
    t = max(0.0, min(1.0, t))
    cos_term = 0.5 * (1.0 + math.cos(math.pi * t))      # 1 -> 0

    return min_lr + (base_lr - min_lr) * cos_term



# ============================================================
# STRATEGIA FEDERATA CUSTOM CON COSINE LR
# ============================================================

class CosineFedAvg(FedAvg):
    """Sottoclasse di FedAvg che aggiorna il learning rate ad ogni round."""

    def __init__(
        self,
        num_rounds: int,
        base_lr: float,
        min_lr: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.base_lr = base_lr
        self.min_lr = min_lr

    def configure_train(self, server_round, arrays, config, grid):
        # Calcola il lr per questo round
        lr_t = cosine_lr_round(
            round_idx=server_round,
            total_rounds=self.num_rounds,
            base_lr=self.base_lr
        )

        # Log comodo
        print(f"[SERVER] Round {server_round}: lr={lr_t:.6e}")

        # Sovrascrivi il valore di lr che sarà visto dai client in cfg["lr"]
        config["lr"] = lr_t

        # Lascia che FedAvg faccia il resto
        return super().configure_train(server_round, arrays, config, grid)


# ============================================================
# WRAPPER PER EVALUATE_FN CON SALVATAGGIO BEST CHECKPOINT
# ============================================================

def make_eval_with_best_checkpoint(
    base_eval_fn,
    best_ckpt_path: str,
):
    best_miou: float = 0.0
    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)

    def eval_with_ckpt(server_round: int, arrays: ArrayRecord):
        nonlocal best_miou

        metric_record = base_eval_fn(server_round, arrays)

        # ⛔ QUI ORA È SBAGLIATO:
        # metrics = metric_record.metrics

        metrics = metric_record  # MetricRecord è un dict-like
        miou = float(metrics.get("server_miou", 0.0))

        if miou > best_miou:
            best_miou = miou
            print(
                f"[SERVER] New BEST checkpoint at round {server_round}: "
                f"mIoU = {best_miou*100:.2f}% -> saving to '{best_ckpt_path}'"
            )
            state_dict = arrays.to_torch_state_dict()
            torch.save(state_dict, best_ckpt_path)

        return metric_record

    return eval_with_ckpt





# ============================================================
# ENTRYPOINT DEL SERVER
# ============================================================

@app.main()
def main(grid: Grid, context: Context) -> None:

    # ------- Seed globale per questa run -------
    seed: int = int(context.run_config.get("seed", 0))
    set_seed(seed)
    
    # ------- Lettura parametri da pyproject.toml -------
    fraction_train: float = float(context.run_config["fraction-train"])
    num_rounds: int = int(context.run_config["num-server-rounds"])
    base_lr: float = float(context.run_config["lr"])
    weight_decay: float = float(context.run_config["weight-decay"])

    # --- NUOVO: local_steps invece di local_epochs ---
    local_steps: int = int(context.run_config["local-steps"])

    try:
        num_total_clients: int = int(context.run_config["num-clients"])
    except KeyError:
        print("Errore: 'num-clients' non trovato nel run_config.")
        return

    print("\n========== SERVER CONFIG ==========")
    print(f"  Seed globale:          {seed}")
    print(f"  Clients totali:        {num_total_clients}")
    print(f"  Round totali:          {num_rounds}")
    print(f"  Base LR:               {base_lr}")
    print(f"  Weight Decay:          {weight_decay}")
    print(f"  Local Steps per round: {local_steps}")
    print(f"  Fraction Train:        {fraction_train}")
    print("===================================\n")

    # ------- Modello globale iniziale -------
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # ------- Funzione di valutazione server-side -------
    device = torch.device("cpu")
    base_server_eval_fn = get_evaluate_fn(device)

    # Path per best checkpoint e modello finale
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(
        ckpt_dir,
        f"non_iid_{local_steps}ls_{num_rounds}tr_{num_total_clients}cl_best_{seed}.pt",
    )
    final_ckpt_path = os.path.join(
        ckpt_dir,
        f"non_iid_{local_steps}ls_{num_rounds}tr_{num_total_clients}cl_last_{seed}.pt",
    )

    # Wrappiamo l'evaluate_fn per salvare il best checkpoint
    server_eval_fn = make_eval_with_best_checkpoint(
        base_eval_fn=base_server_eval_fn,
        best_ckpt_path=best_ckpt_path,
    )

    # ------- Strategia con cosine LR -------
    strategy = CosineFedAvg(
        num_rounds=num_rounds,
        base_lr=base_lr,
        min_lr=1e-6,
        fraction_train=fraction_train,
        min_train_nodes=num_total_clients,      # Sincrono: tutti i client per round
        min_available_nodes=num_total_clients,  # Tutti devono essere disponibili
        fraction_evaluate=0.0,                  # Nessuna valutazione lato client
    )

    # ------- Config base inviata a tutti i client -------
    train_config = ConfigRecord(
        {
            "local_steps": local_steps,
            "lr": base_lr,              # verrà sovrascritto ad ogni round da CosineFedAvg
            "weight-decay": weight_decay,
        }
    )

    print("Avvio training federato...\n")

    # ------- Avvio della federazione -------
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=server_eval_fn,
    )

    # ------- Salvataggio finale -------
    print("\nSaving final (last-round) model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, final_ckpt_path)
    print(f"Modello finale salvato in: {final_ckpt_path}")
    print(f"Miglior modello (best mIoU) salvato in: {best_ckpt_path}")
    print("Federazione completata.")