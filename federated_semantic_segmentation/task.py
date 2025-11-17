"""federated-semantic-segmentation: A Flower / PyTorch app.
   Versione Semplificata (No Augmentation, No Class Weights)
"""

import torch
import torch.nn as nn
# import numpy as np  # <-- Rimosso, non più necessario per i pesi
import torch.nn.functional as F
from federated_semantic_segmentation.utils.mm_model import EarlyFuseLRASPP
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from federated_semantic_segmentation.utils.data import SelmaDrones
from federated_semantic_segmentation.utils.metrics import Metrics
from typing import Dict, Any, Callable
from flwr.app import ArrayRecord, MetricRecord

ROOT_PATH   = "/media/sium/Lexar/syndrone_dataset/renders"
SPLITS_PATH = "/media/sium/Lexar/syndrone_dataset/splits"
SPLIT_KIND  = "iid" # (o "non_iid", a seconda di cosa stai testando)

class Net(nn.Module):
    def __init__(self, num_classes: int = 28, pretrained: bool = True):
        super().__init__()
        self.model = EarlyFuseLRASPP(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)["out"]


# --- MODIFICATO: load_data (rimosso 'augment') ---
def load_data(
    partition_id: int,
    num_partitions: int
):
    """
    Ritorna SOLO il DataLoader di training per il client.
    """
    assert 0 <= partition_id < num_partitions, \
        f"partition_id {partition_id} fuori range (num_partitions={num_partitions})"
    cid = f"C{partition_id+1}"

    bs = 2
    nw = 0
    pin = torch.cuda.is_available()

    # 1. Carica Train Set (Senza 'augment')
    train_set = SelmaDrones(
        root_path=ROOT_PATH,
        splits_path=SPLITS_PATH,
        split_kind=SPLIT_KIND,
        cid=cid,
        server_val=False
        # augment=True <-- RIMOSSO
    )
    print(f"  [Client {cid}] Ricevuto train split: {SPLITS_PATH}/{SPLIT_KIND}/{cid}")
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True,
        drop_last=True, num_workers=nw, pin_memory=pin
    )

    return train_loader


def load_server_data():
    """
    Ritorna UN SINGOLO DataLoader di validazione per il SERVER.
    """
    bs = 2
    nw = 0
    pin = torch.cuda.is_available()
    
    val_set = SelmaDrones(
        root_path=ROOT_PATH,
        splits_path=SPLITS_PATH,
        split_kind=SPLIT_KIND,
        cid=None,
        server_val=True
        # augment=False <-- RIMOSSO
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False,
        drop_last=False, num_workers=nw, pin_memory=pin
    )
        
    return val_loader


# --- MODIFICATO: train (rimossi Class Weights e Grad Clip) ---
def train(net, trainloader, epochs, lr, weight_decay, device):
    """
    Funzione di training standard (senza pesi di classe, senza grad clip).
    """
    import torch
    # import numpy as np # <-- Rimosso

    net.to(device)
    net.train()

    # (Rimossa definizione class_weights_array)
    # (Rimossa definizione class_weights)

    # Criterio standard senza pesi
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=-1
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    # --- DEBUG 2: CONTROLLA SE IL MODELLO È ADDESTRABILE ---
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  [CLIENT DEBUG] Parametri Totali:   {total_params}")
    print(f"  [CLIENT DEBUG] Parametri Addestrabili: {trainable_params}")
    if trainable_params == 0:
        print("  [CLIENT DEBUG] ERRORE: NESSUN PARAMETRO ADDESTRABILE! IL MODELLO È CONGELATO.")
    # --- FINE DEBUG ---

    total_loss = 0.0
    total_pixels = 0

    for _ in range(epochs):
        for (x4, mlb), _meta in trainloader:
            x4  = x4.to(device, dtype=torch.float32)
            mlb = mlb.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = net(x4)
            if isinstance(logits, dict) and "out" in logits:
                logits = logits["out"]

            loss = criterion(logits, mlb)
            loss.backward()

            # (Rimosso) torch.nn.utils.clip_grad_norm_

            optimizer.step()

            valid = (mlb != -1)
            batch_pixels = int(valid.sum().item())
            total_loss  += float(loss.item()) * batch_pixels
            total_pixels += batch_pixels

    avg_train_loss = total_loss / max(1, total_pixels)
    return avg_train_loss


def test(net, testloader, device):
    """
    Funzione di test (invariata, ma usa 'Metrics').
    """
    import torch

    net.to(device)
    net.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)

    cnames = getattr(testloader.dataset, "label_names", None)
    
    metrics = Metrics(cnames, device=device)

    total_loss, total_valid = 0.0, 0

    with torch.no_grad():
        for (x4, mlb), _meta in testloader:
            x4  = x4.to(device, dtype=torch.float32)
            mlb = mlb.to(device, dtype=torch.long)

            logits = net(x4)
            if isinstance(logits, dict) and "out" in logits:
                logits = logits["out"]

            loss = criterion(logits, mlb)

            valid = (mlb != -1)
            valid_pixels = int(valid.sum().item())
            total_loss += float(loss.item()) * valid_pixels
            total_valid += valid_pixels

            preds = logits.argmax(dim=1)
            metrics.add_sample(preds, mlb)

    avg_loss = total_loss / max(1, total_valid)
    miou = float(metrics.percent_mIoU() / 100.0)
    mpa  = float(Metrics.nanmean(100 * metrics.PA()) / 100.0)
    mpp  = float(Metrics.nanmean(100 * metrics.PP()) / 100.0)

    return avg_loss, miou, mpa, mpp


# Incolla questo in task.py (sostituendo la vecchia get_evaluate_fn)

def get_evaluate_fn(
    device: torch.device
) -> Callable[[int, ArrayRecord], MetricRecord]:
    """
    Funzione "Factory" che crea la funzione di valutazione server-side.
    """
    
    print(f"[task.py] Caricamento dataloader server (server_val/test.txt)...")
    val_loader = load_server_data() 
    print(f"[task.py] Dataloader server caricato.")
    
    model = Net().to(device)

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """
        Valuta il modello globale sull'UNICO test set del server.
        """
        print(f"\n--- Esecuzione Server-Side Evaluation (Round {server_round}) ---")
        
        model.load_state_dict(arrays.to_torch_state_dict())
        
        # (test() qui restituisce solo 4 valori, corretto)
        loss, miou, mpa, mpp = test(model, val_loader, device=device)
        
        # !!! MODIFICA QUI: Stampa formattata in percentuale !!!
        print(f"  [SERVER EVAL] Loss: {loss:.4f}, mIoU: {miou * 100:.4f}%")
        
        # (Stampa della tabella rimossa)
        
        metrics_dict: Dict[str, Any] = {
            "server_loss": float(loss),
            "server_miou": float(miou), # Questo rimane 0.0-1.0 per i log
            "server_mpa": float(mpa),
            "server_mpp": float(mpp),
        }

        return MetricRecord(metrics_dict)
    
    return central_evaluate
