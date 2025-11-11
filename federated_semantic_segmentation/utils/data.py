# data.py
# Versione Semplificata + Augmentation (con bugfix sull'ordine)

import cv2 as cv
import numpy as np
from os import path
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import random 
import torchvision.transforms as T
import torchvision.transforms.functional as TF # <-- MODIFICA 1: Aggiunto per la normalizzazione

TARGET_SIZE = (540, 960)  # (H, W)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _norm_rgb(rgb01: np.ndarray) -> np.ndarray:
    # Questa funzione ora è usata solo da denorm_rgb,
    # la normalizzazione principale avviene nel tensore
    return (rgb01 - IMAGENET_MEAN) / IMAGENET_STD

def _norm_depth01(d: np.ndarray) -> np.ndarray:
    return (d - 0.5) / 0.5

# --- MODIFICA 2: to_pytorch ora gestisce la normalizzazione ---
def to_pytorch(rgb_hwc: np.ndarray, depth_hw: np.ndarray, mlb_hw: np.ndarray,
             color_jitter_transform: T.ColorJitter | None = None):
    
    # 1. Converti RGB [0,1] in Tensore
    # rgb_hwc è [H, W, 3] in range [0, 1]
    rgb_t = torch.from_numpy(rgb_hwc).permute(2, 0, 1).contiguous()  # [3,H,W], range [0,1]
    
    # 2. Applica Color Jitter (su range [0,1])
    if color_jitter_transform:
        rgb_t = color_jitter_transform(rgb_t)
    
    # 3. Normalizza (DOPO jitter)
    rgb_t = TF.normalize(rgb_t, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    
    # 4. Concatena
    d1_t  = torch.from_numpy(depth_hw).unsqueeze(0).contiguous()     # [1,H,W], già normalizzato
    x4    = torch.cat([rgb_t, d1_t], dim=0)                          # [4,H,W]
    mlb_t = torch.from_numpy(mlb_hw)
    return x4, mlb_t
# --- FINE MODIFICA 2 ---

def denorm_rgb(norm_rgb_hwc: np.ndarray) -> np.ndarray:
    img = (norm_rgb_hwc * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(img, 0, 1)


class SelmaDrones(Dataset):
    """
    Dataset SynDrone RGB+D+Semantic per training federato.
    Versione Semplificata + Augmentation.
    """
    def __init__(
        self,
        root_path: str,
        splits_path: str,
        split_kind: str,
        cid: Optional[str] = None,
        server_val: bool = False,
        augment: bool = False
    ) -> None:
        assert path.isdir(root_path), f"root_path non valido: {root_path}"
        assert path.isdir(splits_path), f"splits_path non valido: {splits_path}"
        self.root_path = root_path

        if server_val:
            manifest = path.join(splits_path, split_kind, 'server_val', 'test.txt')
        else:
            assert cid is not None, "Specifica 'cid' (es. 'C1') oppure imposta server_val=True"
            manifest = path.join(splits_path, split_kind, cid, 'train.txt')

        assert path.isfile(manifest), f"Manifest non trovato: {manifest}"

        lines = [l.strip() for l in open(manifest, 'r') if l.strip()]
        assert all('/semantic/' in ln for ln in lines), "Tutte le righe devono contenere '/semantic/'"

        self.items = []
        for rel_sem in lines:
            base = path.join(self.root_path, rel_sem.replace('/semantic/', '/%s/')) + ".%s"
            town_tok = rel_sem.split('/')[0]
            town = town_tok.replace('Town', '').split('_')[0]
            htok = [p for p in rel_sem.split('/') if p.startswith('height')]
            height_val = htok[0].replace('height','').replace('m','') if htok else 'unknown'
            self.items.append((base, (height_val, town)))

        assert len(self.items) > 0, "Manifest vuoto dopo il parsing"

        self.label_names = [
            "Building", "Fence", "Other", "Pole", "RoadLine", "Road", "Sidewalk",
            "Vegetation", "Wall", "Traffic Signs", "Sky", "Ground", "Bridge",
            "Rail Track", "Guard Rail", "Traffic Light", "Static", "Dynamic",
            "Water", "Terrain", "Person", "Rider", "Car", "Truck", "Bus",
            "Train", "Motorcycle", "Bicycle"
        ]

        self.idmap = {1:0, 2:1, 3:2, 5:3, 6:4, 7:5, 8:6, 9:7, 11:8, 12:9, 13:10, 14:11, 15:12, 16:13, 17:14,
                      18:15, 19:16, 20:17, 21:18, 22:19, 40:20, 41:21, 100:22, 101:23, 102:24, 103:25, 104:26, 105:27}
        
        self.cmap = np.array([
            [ 70, 70, 70], # building
            [190,153,153], # fence
            [180,220,135], # other
            [153,153,153], # pole
            [255,255,255], # road line
            [128, 64,128], # road
            [244, 35,232], # sidewalk
            [107,142, 35], # vegetation
            [102,102,156], # wall
            [220,220,  0], # traffic sign
            [ 70,130,180], # sky
            [ 81,  0, 81], # ground
            [150,100,100], # bridge
            [230,150,140], # rail track
            [180,165,180], # guard rail
            [250,170, 30], # traffic light
            [110,190,160], # static
            [111, 74,  0], # dynamic
            [ 45, 60,150], # water
            [152,251,152], # terrain
            [220, 20, 60], # person
            [255,  0,  0], # rider
            [  0,  0,142], # car
            [  0,  0, 70], # truck
            [  0, 60,100], # bus
            [  0, 80,100], # train
            [  0,  0,230], # motorcycle
            [119, 11, 32], # bicycle
            [  0,  0,  0], # unknown
        ], dtype=np.uint8)

        self.augment = augment
        self.color_jitter = None
        if self.augment:
            self.color_jitter = T.ColorJitter(brightness=0.6, hue=0.4)
            print(f"ATTENZIONE: Augmentation (Flip, ColorJitter) ATTIVATA per il dataset.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        fbase, (h,t) = self.items[idx]
        
        # Carica i dati grezzi (ancora array NumPy)
        rgb = cv.imread(fbase % ('rgb', 'jpg'), cv.IMREAD_UNCHANGED)[..., ::-1]
        rgb = rgb.astype(np.float32) / 255.0 # Range [0, 1]
        dth = cv.imread(fbase % ('depth', 'png'), cv.IMREAD_UNCHANGED).astype(np.float32)
        dth = dth / (256 * 256 - 1) # Range [0, 1]
        sem = cv.imread(fbase % ('semantic', 'png'), cv.IMREAD_UNCHANGED)
        
        # Applica Augmentation Geometriche (Flipping)
        if self.augment:
            if random.random() < 0.5:
                rgb = np.fliplr(rgb).copy()
                dth = np.fliplr(dth).copy()
                sem = np.fliplr(sem).copy()

        # Resize (DOPO augmentation)
        rgb = cv.resize(rgb, TARGET_SIZE, interpolation=cv.INTER_LINEAR)
        dth = cv.resize(dth, TARGET_SIZE, interpolation=cv.INTER_NEAREST)
        sem = cv.resize(sem, TARGET_SIZE, interpolation=cv.INTER_NEAREST)

        # --- MODIFICA 3: Ordine di Normalizzazione Corretto ---
        # 1. Normalizza Depth (NumPy)
        dth = _norm_depth01(dth) 
        
        # 2. NON normalizzare RGB qui. Rimane in range [0, 1]
        # rgb = _norm_rgb(rgb) # <-- RIMOSSO (questo era il bug)
        # --- FINE MODIFICA 3 ---

        # Creazione Maschera
        mlb = -1 * np.ones_like(sem, dtype=np.int32)
        for k, v in self.idmap.items():
            mlb[sem == k] = v

        # Pacchetto finale
        # to_pytorch ora gestisce ColorJitter E normalizzazione RGB
        x4, mlb_t = to_pytorch(rgb, dth, mlb, self.color_jitter) 
        return (x4, mlb_t), (h, t)

    def color_label(self, label_hw: np.ndarray) -> np.ndarray:
        # ... (codice invariato) ...
        cmap_with_ignore = self.cmap
        if len(self.cmap) == 28:
             cmap_with_ignore = np.vstack([self.cmap, [0, 0, 0]])
        label_hw = label_hw.copy()
        label_hw[label_hw == -1] = 28
        return cmap_with_ignore[label_hw.astype(int)]

    def scale_depth(self, ts):
        # ... (codice invariato) ...
        ts[ts<0] = -np.log10(-ts[ts<0])
        ts[ts>0] = np.log10(ts[ts>0])
        return ts

        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    print("Esecuzione smoke test data.py...")
    d = SelmaDrones(
        root_path="/media/sium/Lexar/syndrone_dataset/renders",
        splits_path="/media/sium/Lexar/syndrone_dataset/splits",
        split_kind="non_iid", # Esempio
        cid="C1",
        server_val=False,
        augment=True # <-- Test con augmentation attiva
    )
    (x4, mlb), _ = d[0]
    fig, axs = plt.subplots(1,2, figsize=(16, 8))
    
    # x4 è normalizzato, denorm_rgb è necessario
    rgb_vis = denorm_rgb(x4[:3].permute(1,2,0).numpy())
    axs[0].imshow(rgb_vis)
    axs[0].set_title("Input RGB (Augmented)")
    
    axs[1].imshow(d.color_label(mlb.numpy()))
    axs[1].set_title("Label (Augmented)")
    
    print(f"Campione 0 caricato (x4 shape: {x4.shape}, mlb shape: {mlb.shape})")
    plt.show()