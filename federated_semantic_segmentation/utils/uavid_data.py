# uavid_data.py
# Dataset per UAVid (valutazione su immagini reali)
# Usa lo stesso schema di input del modello: RGB normalizzato + depth fittizia + altitudine scalare.

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from os import path
from glob import glob
from typing import List, Tuple
import torchvision.transforms.functional as TF

# Stessa risoluzione usata per SynDrone
TARGET_SIZE = (768, 432)  # (W, H) per cv2

# Normalizzazione ImageNet (come nel tuo data.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _norm_depth01(d: np.ndarray) -> np.ndarray:
    """Normalizza un depth in [0,1] in [-1,1] (stesso schema usato per SynDrone)."""
    return (d - 0.5) / 0.5


def to_pytorch(rgb_hwc: np.ndarray, depth_hw: np.ndarray, lbl_hw: np.ndarray):
    """
    Converte RGB [0,1], depth normalizzato e label in tensori PyTorch.
    RGB viene normalizzato con mean/std ImageNet.
    """
    # RGB: [H,W,3] float32 [0,1] -> [3,H,W]
    rgb_t = torch.from_numpy(rgb_hwc).permute(2, 0, 1).contiguous()
    rgb_t = TF.normalize(rgb_t, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())

    # Depth: [H,W] -> [1,H,W]
    d1_t = torch.from_numpy(depth_hw).unsqueeze(0).contiguous()

    # Concatena: [4,H,W]
    x4 = torch.cat([rgb_t, d1_t], dim=0)

    # Label: [H,W]
    lbl_t = torch.from_numpy(lbl_hw.astype(np.int64))

    return x4, lbl_t


class UAVidDataset(Dataset):
    """
    Dataset per la valutazione su UAVid (immagini reali).
    
    Assunzioni:
    - img_dir contiene immagini RGB (.png/.jpg) rinominate 000000.png, 000001.png, ...
    - label_dir contiene le corrispondenti maschere a 3 canali (palette) o a 1 canale.
    - Le classi 0..7 sono lo spazio coarse:
        0: Road
        1: Nature
        2: Person
        3: Vehicle
        4: Construction
        5: Obstacle
        6: Water
        7: Void
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        subset_name: str = "clean",
        fake_altitude_m: float = 50.0,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:

        assert path.isdir(img_dir), f"img_dir non valido: {img_dir}"
        assert path.isdir(label_dir), f"label_dir non valido: {label_dir}"

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.subset_name = subset_name
        self.fake_altitude_m = float(fake_altitude_m)

        # Trova tutte le immagini supportate e ordinale
        img_paths: List[str] = []
        for ext in extensions:
            img_paths.extend(glob(path.join(img_dir, f"*{ext}")))
        img_paths = sorted(img_paths)

        assert len(img_paths) > 0, f"Nessuna immagine trovata in {img_dir}"

        # Costruisce la lista (img_path, label_path)
        self.items: List[Tuple[str, str]] = []
        for ip in img_paths:
            fname = path.splitext(path.basename(ip))[0]  # "000000"
            # si assume che la label abbia lo stesso nome e sia .png
            lp = path.join(label_dir, fname + ".png")
            if not path.isfile(lp):
                raise FileNotFoundError(f"Label mancante per {ip}: {lp}")
            self.items.append((ip, lp))

        # -----------------------------
        # Spazio delle etichette coarse (8 classi)
        # -----------------------------
        self.coarse_labels = [
            "Road",         # 0
            "Nature",       # 1
            "Person",       # 2
            "Vehicle",      # 3
            "Construction", # 4
            "Obstacle",     # 5
            "Water",        # 6
            "Void",         # 7
        ]

        # Colormap per le 8 classi coarse (colori che mi hai fornito)
        self.cmap_coarse = np.array([
            [130,  64, 129],  # Road
            [107, 143,  60],  # Nature
            [221,  29,  61],  # Person
            [ 42,  45, 126],  # Vehicle
            [ 69,  70,  70],  # Construction
            [153, 153, 153],  # Obstacle
            [ 43,  62, 150],  # Water
            [  4,   5,   3],  # Void
        ], dtype=np.uint8)

        self.coarse_void_id = 7

        # Mappa colore (BGR) -> ID coarse [0..7]
        # Palette UAVid (BGR):
        #   (  0,   0,   0): background clutter
        #   (  0,   0, 128): building
        #   (  0,  64,  64): human
        #   (  0, 128,   0): tree
        #   (  0, 128, 128): low vegetation
        #   (128,   0,  64): static car
        #   (128,  64, 128): road
        #   (192,   0, 192): moving car

        self.color_to_id = {
            (  0,   0,   0): 7,  # background clutter -> Void
            (  0,   0, 128): 4,  # building           -> Construction
            (  0,  64,  64): 2,  # human              -> Person
            (  0, 128,   0): 1,  # tree               -> Nature
            (  0, 128, 128): 1,  # low vegetation     -> Nature
            (128,   0,  64): 3,  # static car         -> Vehicle
            (128,  64, 128): 0,  # road               -> Road
            (192,   0, 192): 3,  # moving car         -> Vehicle
        }


    def __len__(self) -> int:
        return len(self.items)

    def _label_rgb_to_id(self, lbl_rgb: np.ndarray) -> np.ndarray:
        """
        Converte una mask RGB (palette) [H,W,3] in una mask [H,W] con ID 0..7
        usando self.color_to_id (in BGR).
        """
        h, w, _ = lbl_rgb.shape
        lbl_id = np.full((h, w), fill_value=self.coarse_void_id, dtype=np.int32)

        # Attenzione: lbl_rgb è BGR (OpenCV)
        b = lbl_rgb[:, :, 0]
        g = lbl_rgb[:, :, 1]
        r = lbl_rgb[:, :, 2]

        for (cb, cg, cr), cid in self.color_to_id.items():
            mask = (b == cb) & (g == cg) & (r == cr)
            lbl_id[mask] = cid

        return lbl_id

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.items[idx]
        fname = path.splitext(path.basename(img_path))[0]

        # -----------------------------
        # Carica RGB
        # -----------------------------
        rgb = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        if rgb is None:
            raise RuntimeError(f"Impossibile leggere immagine: {img_path}")
        if rgb.ndim == 2:
            rgb = cv.cvtColor(rgb, cv.COLOR_GRAY2RGB)
        else:
            rgb = rgb[..., ::-1]  # BGR -> RGB

        rgb = rgb.astype(np.float32) / 255.0

        # -----------------------------
        # Carica label (palette RGB) e convertila in ID 0..7
        # -----------------------------
        lbl = cv.imread(lbl_path, cv.IMREAD_UNCHANGED)
        if lbl is None:
            raise RuntimeError(f"Impossibile leggere label: {lbl_path}")

        if lbl.ndim == 2:
            # Caso raro: già ID
            lbl_id = lbl.astype(np.int32)
        else:
            # lbl è BGR a 3 canali: usiamo la LUT
            lbl_id = self._label_rgb_to_id(lbl)

        # -----------------------------
        # Resize
        # -----------------------------
        rgb = cv.resize(rgb, TARGET_SIZE, interpolation=cv.INTER_AREA)
        lbl_id = cv.resize(lbl_id, TARGET_SIZE, interpolation=cv.INTER_NEAREST)

        # Clippa/normalizza gli ID fuori range in Void
        lbl_id[(lbl_id < 0) | (lbl_id > self.coarse_void_id)] = self.coarse_void_id

        # -----------------------------
        # Depth fittizia + normalizzazione
        # -----------------------------
        depth = np.full((TARGET_SIZE[1], TARGET_SIZE[0]), 0.5, dtype=np.float32)
        depth = _norm_depth01(depth)  # -> 0.0

        # -----------------------------
        # Conversione in tensori
        # -----------------------------
        x4, lbl_t = to_pytorch(rgb, depth, lbl_id)

        # Altitudine fittizia (in metri), come tensore [1]
        alt_t = torch.tensor(self.fake_altitude_m, dtype=torch.float32)

        # Meta-info: utile per debug / salvataggio
        meta = {
            "filename": fname,
            "img_path": img_path,
            "lbl_path": lbl_path,
            "subset": self.subset_name,
            "altitude_m": self.fake_altitude_m,
        }

        return x4, lbl_t, alt_t, meta

    # ===========================
    # Visualizzazione coarse
    # ===========================
    def color_label_coarse(self, coarse_hw: np.ndarray) -> np.ndarray:
        """
        Converte una mask [H,W] con ID 0..7 in immagine RGB usando la colormap coarse.
        """
        lab = coarse_hw.copy()
        lab[lab < 0] = self.coarse_void_id
        lab[lab >= len(self.cmap_coarse)] = self.coarse_void_id
        return self.cmap_coarse[lab.astype(int)]
