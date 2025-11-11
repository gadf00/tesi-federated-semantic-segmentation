import torch

# Importa le classi per la nuova API serverapp
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

# Importa le tue funzioni da task.py
# (get_evaluate_fn ora non ha più 'run_full_experiment')
from federated_semantic_segmentation.task import Net, get_evaluate_fn

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    
    # --- Run config ---
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    weight_decay: float = context.run_config["weight-decay"]
    local_epochs: int = context.run_config["local-epochs"]
    
    # --- MODIFICA: Rimosso 'run_full_experiment' ---
    # run_full_experiment: bool = bool(context.run_config.get("run-full-experiment", False))
    
    try:
        num_total_clients: int = int(context.run_config["num-clients"])
    except KeyError:
        print("Errore: 'num-clients' non trovato nel run_config. Assicurati sia in pyproject.toml")
        return
    
    device = torch.device("cpu")

    # --- Modello globale iniziale ---
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # --- Strategy (FedAvg standard) ---
    # Questo è corretto:
    strategy = FedAvg(
        fraction_train=fraction_train,
        min_train_nodes=num_total_clients,
        fraction_evaluate=0.0, # Disabilita evaluation lato client
    )
    
    # --- Prepara la funzione di valutazione server-side ---
    # --- MODIFICA: 'get_evaluate_fn' ora richiede solo 'device' ---
    server_eval_fn = get_evaluate_fn(device)

    print("Avvio training federato...")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "lr": lr,
            "weight-decay": weight_decay,
            "local_epochs": local_epochs,
        }),
        num_rounds=num_rounds,
        evaluate_fn=server_eval_fn,
    )

    # --- Salvataggio finale ---
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    print("Federazione completata.")