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

ROOT_PATH   = "C://Users/Lab/Desktop/syndrone_dataset/renders"
SPLITS_PATH = "C://Users/Lab/Desktop/syndrone_dataset/splits"
SPLIT_KIND  = "non_iid" # (o "non_iid", a seconda di cosa stai testando)

class Net(nn.Module):
    def __init__(self, num_classes: int = 28, pretrained: bool = True):
        super().__init__()
        self.model = EarlyFuseLRASPP(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x4, altitude):
        # EarlyFuseLRASPP già ritorna un dict {"out": logits}
        out = self.model(x4, altitude)
        if isinstance(out, dict) and "out" in out:
            return out["out"]
        return out



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

    # 1. Carica Train Set
    train_set = SelmaDrones(
        root_path=ROOT_PATH,
        splits_path=SPLITS_PATH,
        split_kind=SPLIT_KIND,
        cid=cid,
        server_val=False
    )
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
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False,
        drop_last=False, num_workers=nw, pin_memory=pin
    )
        
    return val_loader

def train(net, trainloader, local_steps, lr, weight_decay, device):
    net.to(device)
    net.train()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    total_pixels = 0
    steps_done = 0

    data_iter = iter(trainloader)

    while steps_done < local_steps:
        try:
            x4, mlb, alt, _meta = next(data_iter)
        except StopIteration:
            data_iter = iter(trainloader)
            x4, mlb, alt, _meta = next(data_iter)

        x4  = x4.to(device, dtype=torch.float32)   # [B,4,H,W]
        mlb = mlb.to(device, dtype=torch.long)     # [B,H,W]
        alt = alt.to(device, dtype=torch.float32)  # [B]

        optimizer.zero_grad()

        logits = net(x4, alt)
        loss = criterion(logits, mlb)

        loss.backward()
        optimizer.step()

        valid = (mlb != -1)
        batch_pixels = int(valid.sum().item())
        total_loss += float(loss.item()) * batch_pixels
        total_pixels += batch_pixels

        steps_done += 1

    avg_train_loss = total_loss / max(1, total_pixels)
    return avg_train_loss


def test(net, testloader, device):
    net.to(device)
    net.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)

    cnames = getattr(testloader.dataset, "label_names", None)
    metrics = Metrics(cnames, device=device)

    total_loss, total_valid = 0.0, 0

    with torch.no_grad():
        for x4, mlb, alt, _meta in testloader:
            x4  = x4.to(device, dtype=torch.float32)
            mlb = mlb.to(device, dtype=torch.long)
            alt = alt.to(device, dtype=torch.float32)

            logits = net(x4, alt)
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



def get_evaluate_fn(
    device: torch.device
) -> Callable[[int, ArrayRecord], MetricRecord]:
    
    val_loader = load_server_data()
    model = Net().to(device)

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        sd = arrays.to_torch_state_dict()
        model.load_state_dict(sd)

        loss, miou, mpa, mpp = test(model, val_loader, device=device)
        print(f"  [SERVER EVAL] Loss: {loss:.4f}, mIoU: {miou * 100:.4f}%")

        metrics_dict: Dict[str, Any] = {
            "server_loss": float(loss),
            "server_miou": float(miou),
            "server_mpa": float(mpa),
            "server_mpp": float(mpp),
        }

        return MetricRecord(metrics_dict)
    
    return central_evaluate
