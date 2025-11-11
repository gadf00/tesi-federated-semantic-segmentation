# mm_model_gn.py
# Early-fusion RGB+D LR-ASPP MobileNetV3 with optional GroupNorm conversion
# and correct transfer learning (pre-trained weights + head replacement).

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
        # Carica i pesi pre-addestrati su COCO
        return LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
except Exception:  # older torchvision
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large  # type: ignore
    def _get_weights(pretrained: bool):
        # Vecchio metodo per caricare i pesi
        return "DEFAULT" if pretrained else None


def _first_conv2d(module: nn.Module) -> nn.Conv2d:
    """Find first Conv2d in backbone to patch 3->4 channels."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d found in backbone")


def _patch_rgb_to_rgbd(first_conv: nn.Conv2d, init_depth: str = "zeros") -> None:
    """
    Patches the first Conv2d layer (3 channels) to accept a 4th channel (Depth).
    It preserves the pre-trained weights for the RGB channels.
    """
    assert first_conv.in_channels == 3, "Unexpected first conv: not 3 input channels"
    with torch.no_grad():
        w3 = first_conv.weight  # [Cout, 3, k, k]
        Cout, _, k1, k2 = w3.shape
        # Crea un nuovo tensore di pesi per 4 canali
        w4 = torch.zeros(Cout, 4, k1, k2, device=w3.device, dtype=w3.dtype)
        # Copia i pesi RGB originali
        w4[:, :3] = w3
        # Inizializza i pesi del 4° canale (Depth)
        if init_depth == "mean":
            w4[:, 3] = w3.mean(dim=1)
        # (Se init_depth == "zeros", i pesi rimangono 0)
        
        # Assegna i nuovi pesi
        first_conv.in_channels = 4
        first_conv.weight = nn.Parameter(w4)
        # Il bias (se presente) rimane invariato


def convert_bn_to_gn(module: nn.Module, num_groups: int = 16) -> nn.Module:
    """Recursively replace BatchNorm2d with GroupNorm.
    Ensures channels % groups == 0 by reducing groups when needed.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            g = min(num_groups, C)
            while g > 1 and (C % g) != 0:
                g -= 1
            gn = nn.GroupNorm(num_groups=g, num_channels=C, affine=True)
            # Copia i pesi (gamma) e i bias (beta) se l'affine è True
            if gn.affine:
                gn.weight.data = child.weight.data.clone().detach()
                gn.bias.data = child.bias.data.clone().detach()
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)
    return module


def freeze_backbone(model: nn.Module, requires_grad: bool = False) -> None:
    """Enable/disable grads for the backbone (feature extractor)."""
    if hasattr(model, "net") and hasattr(model.net, "backbone"):
        for p in model.net.backbone.parameters():
            p.requires_grad = requires_grad


def unfreeze_backbone(model: nn.Module) -> None:
    freeze_backbone(model, requires_grad=True)


class EarlyFuseLRASPP(nn.Module):
    """
    Early-fusion RGB+D (4 channels) LR-ASPP with MobileNetV3 backbone.
    - Carica correttamente i pesi pre-addestrati (Transfer Learning).
    - Sostituisce la testa di classificazione per il numero di classi desiderato.
    - Converte opzionalmente BatchNorm in GroupNorm (ideale per FL e Batch Size 1).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        init_depth: str = "zeros",
        use_gn: bool = True,
        gn_groups: int = 16,
    ) -> None:
        super().__init__()
        
        weights = _get_weights(pretrained)
        
        # 1. Carica il modello pre-addestrato 
        #    Usare le classi originali (21) per caricare correttamente i pesi.
        original_num_classes = 21 
        self.net = lraspp_mobilenet_v3_large(
            num_classes=original_num_classes,
            weights=weights
        )

        # 2. Applica la patch 3->4 canali (RGB-D) sul backbone pre-addestrato
        first_conv = _first_conv2d(self.net.backbone)
        _patch_rgb_to_rgbd(first_conv, init_depth=init_depth)

        # 3. Sostituisci la testa di classificazione (da 21 a num_classes)
        try:
            # Recupera i canali di input dalla vecchia testa
            # accedendo ai layer per NOME, non per indice
            high_channels = self.net.classifier.cbr[0].in_channels
            low_channels = self.net.classifier.low_classifier.in_channels
            inter_channels = self.net.classifier.cbr[0].out_channels # o .scale[1].out_channels

            # Crea la NUOVA testa con il tuo num_classes
            self.net.classifier = LRASPPHead(
                low_channels=low_channels,
                high_channels=high_channels,
                num_classes=num_classes,
                inter_channels=inter_channels
            )
        except Exception as e:
            print(f"ERRORE: Impossibile sostituire la testa del classificatore: {e}")
            print("La struttura di LRASPP in torchvision potrebbe essere cambiata.")
            raise

        # 4. Converti BN -> GN (per stabilità con Batch Size 1 in FL)
        #    Questo va fatto DOPO aver creato la nuova testa,
        #    perché anche la testa LRASPPHead contiene layer BatchNorm.
        if use_gn:
            convert_bn_to_gn(self.net, num_groups=gn_groups)
        
        self.num_classes = num_classes

    def forward(self, x4: torch.Tensor):
        # x4 ha dimensioni [N, 4, H, W]
        return self.net(x4)


# Backwards-compatible alias
EarlyFuse = EarlyFuseLRASPP