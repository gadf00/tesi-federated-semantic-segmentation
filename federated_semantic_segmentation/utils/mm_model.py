# mm_model.py
# Early-fusion RGB+D+Alt LR-ASPP MobileNetV3 
# (Senza conversione GroupNorm)

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

# Import necessario per ricostruire la testa del classificatore
from torchvision.models.segmentation.lraspp import LRASPPHead

# --- Robust import of weights enum across torchvision versions ---
try:  # torchvision >= 0.13
    from torchvision.models.segmentation import (
        lraspp_mobilenet_v3_large,
        LRASPP_MobileNet_V3_Large_Weights,
    )

    def _get_weights(pretrained: bool):
        # Carica i pesi pre-addestrati su COCO/VOC
        return (
            LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            if pretrained
            else None
        )

except Exception:  # older torchvision
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large  # type: ignore

    def _get_weights(pretrained: bool):
        # Vecchio metodo per caricare i pesi
        return "DEFAULT" if pretrained else None


def _first_conv2d(module: nn.Module) -> nn.Conv2d:
    """Find first Conv2d in backbone to patch."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d found in backbone")


def _patch_rgb_to_rgbdalt(first_conv: nn.Conv2d, init_depth: str = "zeros") -> None:
    """
    Patch del primo Conv2d (3 canali RGB) per accettare 5 canali:
    [R, G, B, Depth, Altitude].

    - I pesi RGB pre-addestrati vengono preservati.
    - Il canale Depth (4°) viene inizializzato a:
        * 0 se init_depth == "zeros"
        * media dei 3 canali RGB se init_depth == "mean"
    - Il canale Altitude (5°) viene inizializzato a zero.
    """
    assert first_conv.in_channels == 3, "Unexpected first conv: not 3 input channels"
    with torch.no_grad():
        w3 = first_conv.weight  # [Cout, 3, k, k]
        Cout, _, k1, k2 = w3.shape

        # Nuovo tensore pesi: 5 canali (RGB + Depth + Altitude)
        w5 = torch.zeros(Cout, 5, k1, k2, device=w3.device, dtype=w3.dtype)

        # Copia i pesi originali RGB
        w5[:, :3] = w3

        # Inizializza il canale Depth
        if init_depth == "mean":
            # Media sui 3 canali RGB
            w5[:, 3] = w3.mean(dim=1)
        # Se "zeros", il canale Depth resta a 0

        # Il 5° canale (Altitude) rimane a zero

        # Assegna i nuovi pesi
        first_conv.in_channels = 5
        first_conv.weight = nn.Parameter(w5)
        # Il bias (se presente) rimane invariato


def freeze_backbone(model: nn.Module, requires_grad: bool = False) -> None:
    """Enable/disable grads for the backbone (feature extractor)."""
    if hasattr(model, "net") and hasattr(model.net, "backbone"):
        for p in model.net.backbone.parameters():
            p.requires_grad = requires_grad


def unfreeze_backbone(model: nn.Module) -> None:
    freeze_backbone(model, requires_grad=True)


class EarlyFuseLRASPP(nn.Module):
    """
    Early-fusion RGB + Depth + Altitude con LR-ASPP (MobileNetV3 backbone).

    Input:
        x4:       [N, 4, H, W]  -> [R,G,B,D] (RGB già normalizzati + depth in [-1,1])
        altitude: [N] oppure [N,1]  -> quota in metri (es. 20, 50, 80)

    Internamente:
        - Il primo Conv2d del backbone viene patchato 3 -> 5 canali.
        - Il 5° canale è una mappa costante dell'altitudine normalizzata:
            alt_norm = (altitude - alt_mean) / alt_std
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        init_depth: str = "zeros",
        alt_mean: float = 50.0,   # media tipica per {20,50,80}
        alt_std: float = 30.0,    # deviazione std approssimativa
    ) -> None:
        super().__init__()

        self.alt_mean = alt_mean
        self.alt_std = alt_std
        assert init_depth in ("zeros", "mean"), "init_depth must be 'zeros' or 'mean'"

        weights = _get_weights(pretrained)

        # 1. Carica il modello pre-addestrato.
        #    Usare le classi originali (21) per caricare correttamente i pesi.
        original_num_classes = 21
        self.net = lraspp_mobilenet_v3_large(
            num_classes=original_num_classes,
            weights=weights,
        )

        # 2. Patch 3 -> 5 canali (RGB + Depth + Altitude) sul backbone pre-addestrato
        first_conv = _first_conv2d(self.net.backbone)
        _patch_rgb_to_rgbdalt(first_conv, init_depth=init_depth)

        # 3. Sostituisci la testa di classificazione (da 21 a num_classes)
        try:
            # Recupera i canali dalla testa esistente
            high_channels = self.net.classifier.cbr[0].in_channels
            low_channels = self.net.classifier.low_classifier.in_channels
            inter_channels = self.net.classifier.cbr[0].out_channels

            # Crea la NUOVA testa con il tuo num_classes
            self.net.classifier = LRASPPHead(
                low_channels=low_channels,
                high_channels=high_channels,
                num_classes=num_classes,
                inter_channels=inter_channels,
            )
        except Exception as e:
            print(f"ERRORE: Impossibile sostituire la testa del classificatore: {e}")
            print("La struttura di LRASPP in torchvision potrebbe essere cambiata.")
            raise

        self.num_classes = num_classes

    def forward(self, x4: torch.Tensor, altitude: torch.Tensor):
        """
        x4:       [N, 4, H, W]  -> RGBD
        altitude: [N] oppure [N,1] -> quota in metri
        """
        N, C, H, W = x4.shape
        assert C == 4, f"Expected 4 input channels (RGBD), got {C}"

        # altitude -> [N,1,1,1]
        if altitude.dim() == 1:
            altitude = altitude.view(N, 1, 1, 1)
        elif altitude.dim() == 2 and altitude.shape[1] == 1:
            altitude = altitude.view(N, 1, 1, 1)
        else:
            raise ValueError("altitude must be [N] or [N,1]")

        # Normalizza altitudine
        alt_norm = (altitude - self.alt_mean) / (self.alt_std + 1e-6)
        alt_map = alt_norm.expand(-1, 1, H, W)  # [N,1,H,W]

        # Costruisci input a 5 canali: [R,G,B,D,Alt]
        x5 = torch.cat([x4, alt_map], dim=1)  # [N,5,H,W]

        return self.net(x5)


# Backwards-compatible alias
EarlyFuse = EarlyFuseLRASPP