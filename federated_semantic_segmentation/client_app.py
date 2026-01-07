from typing import Dict, Any
import torch
import random
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import time

from federated_semantic_segmentation.task import (
    Net, load_data, train as train_fn
)

app = ClientApp()

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for this client."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Esegue il training locale usando local_steps.
    Restituisce i pesi aggiornati e la loss locale.
    """

    # 1) Config del round (inviata dal Server)
    cfg: Dict[str, Any] = msg.content.get("config", {})

    # Learning rate e weight decay aggiornati dal server
    lr: float = float(cfg.get("lr", 1e-3))
    weight_decay: float = float(cfg.get("weight-decay", 0.01))


    local_steps: int = int(cfg.get("local_steps", 300))

    # 1.5) Seed per questo client
    global_seed = int(context.run_config.get("seed", 0))
    cid0 = int(context.node_config["partition-id"]) 
    client_seed = global_seed + cid0
    set_seed(client_seed)

    # 2) Carica il modello globale
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) Dataset locale
    cid0 = int(context.node_config["partition-id"])  # 0,1,2
    num_partitions = context.node_config["num-partitions"]
    cid1 = cid0 + 1                                  # per print umane
    trainloader = load_data(cid0, num_partitions)

    print(f"  [Client {cid1}] Inizio training locale... (local_steps={local_steps}, LR={lr})")
    start_time = time.time()
    # 4) Training locale
    train_loss: float = train_fn(model, trainloader, local_steps, lr, weight_decay, device)
    num_train_examples = len(trainloader.dataset)
    elapsed = time.time() - start_time
    print(f"  [Client {cid1}] Fine training locale. Loss: {train_loss:.4f} (time: {elapsed:.2f}s)")

    # 5) Risposta
    metrics_dict: Dict[str, Any] = {
        "cid": cid0,
        "train_loss": float(train_loss),

        # FedAvg richiede sempre 'num-examples'
        "num-examples": num_train_examples,
    }

    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord(metrics_dict)
    content = RecordDict({"arrays": arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)



@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Valutazione disabilitata (il server usa fraction_evaluate=0.0).
    """

    cid0 = int(context.node_config["partition-id"])
    print(f"Client C{cid0+1}: @app.evaluate() chiamata, ma non implementata (OK).")

    # Risposta minima per non disturbare FedAvg
    metrics = MetricRecord({
        "cid": cid0,
        "eval_loss": float("nan"),
        "eval_miou": float("nan"),
        "num-examples": 0,
    })
    
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)