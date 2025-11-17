"""Client application for federated semantic segmentation (Flower 1.23).
   Versione Semplificata: solo training.
"""

from typing import Dict, Any
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
# --- MODIFICA CHIAVE ---
# Importiamo solo Net, load_data, e train_fn (NON test_fn)
from federated_semantic_segmentation.task import (
    Net, load_data, train as train_fn
)

# Crea l'app client
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Esegue SOLO il training locale e restituisce i pesi
    aggiornati e la loss di training.
    """
    
    # 1) Config round
    cfg: Dict[str, Any] = msg.content.get("config", {})
    # 'run_full_experiment' rimosso

    # 2) Modello e pesi globali
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) Dati locali
    cid0 = int(context.node_config["partition-id"])
    num_partitions = context.node_config["num-partitions"]
    cid1 = cid0 + 1
    trainloader = load_data(
        cid0, 
        num_partitions
    )

    # 4) Hyper locali
    lr: float = float(cfg.get("lr", 1e-3))
    weight_decay: float = float(cfg.get("weight-decay", 0.01))
    local_epochs: int = int(cfg.get("local_epochs", 1))

    # --- DEBUG 1: CONTROLLA IL LR REALE ---
    print(f"  [CLIENT DEBUG {cid1}] LR in uso: {lr}")
    if lr == 0.0:
        print(f"  [CLIENT DEBUG {cid1}] ERRORE: LR È ZERO!")
    # --- FINE DEBUG ---

    print(f"  [Client {cid1}] Inizio training locale... (Epochs: {local_epochs}, LR: {lr})")

    # 5) Training locale
    train_loss: float = train_fn(model, trainloader, local_epochs, lr, weight_decay, device)
    num_train_examples = len(trainloader.dataset)

    print(f"  [Client {cid1}] Fine training locale. Loss: {train_loss:.4f}")

    # 7) Prepara la risposta 
    metrics_dict: Dict[str, Any] = {
        "cid": cid0,
        "train_loss": float(train_loss),
        # La chiave 'num-examples' è obbligatoria per FedAvg
        "num-examples": num_train_examples, 
    }
    
    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord(metrics_dict)
    content = RecordDict({"arrays": arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Funzione di valutazione (opzionale, "vuota").
    Non verrà chiamata se il server imposta fraction_evaluate=0.0.
    """

    cid0 = int(context.node_config["partition-id"])
    print(f"Client C{cid0+1}: @app.evaluate() chiamata, ma non implementata (OK).")

    # Restituisce metriche vuote per non bloccare
    metrics = MetricRecord({
        "cid": cid0, 
        "eval_loss": float("nan"),
        "eval_miou": float("nan"),
        # Aggiungiamo 'num-examples' anche qui per sicurezza
        "num-examples": 0
    })
    
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)