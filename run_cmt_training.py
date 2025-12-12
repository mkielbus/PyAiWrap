import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from pyaiwrap.utils import prepareDevice
from pyaiwrap.neural_network import ColorMemoryTransformer


# =======================
# 1. Dataset
# =======================

class GrayscaleAutoencodingDataset(Dataset):
    """
    Bardzo prosty dataset:
    - czyta kolorowe obrazy z folderu
    - zamienia je na skalę szarości
    - zwraca (wejście_L, target_L), czyli to samo (autoenkoder)
    """

    def __init__(self, root_dir: str, image_size: int = 256):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        self.image_paths: List[str] = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f.lower())[1] in exts
        ]

        if not self.image_paths:
            raise RuntimeError(f"Brak obrazów w folderze: {root_dir}")

        # transformacja: zmiana rozmiaru + tensor + konwersja do szarości
        self.transform_color = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),              # [C, H, W], 0..1
        ])
        self.to_gray = transforms.Grayscale(num_output_channels=1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        img = self.transform_color(img)        # [3, H, W]
        gray = self.to_gray(img)              # [1, H, W]

        # wejście = target (autoenkoder jasności)
        return gray, gray


# =======================
# 2. Trening ColorMemoryTransformer
# =======================

def train_cmt(
    train_dir: str = "data/train",
    val_dir: str = "data/val",
    image_size: int = 128,      # MNIEJSZE OBRAZY -> mniej pamięci
    batch_size: int = 2,        # mniejszy batch na CPU
    num_epochs: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    save_dir: str = "weights"
):
    device = prepareDevice(use_cuda=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- datasets & dataloaders ---
    train_ds = GrayscaleAutoencodingDataset(train_dir, image_size=image_size)
    val_ds = GrayscaleAutoencodingDataset(val_dir, image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # --- model (LŻEJSZY) ---
    # zmniejszamy embed_dim, num_heads, memory_size
    from pyaiwrap.neural_network import ColorMemoryTransformer

    model = ColorMemoryTransformer(
        embed_dim=256,      # było 512
        num_heads=4,        # było 8
        mlp_ratio=4,
        dropout=0.1,
        color_decoder_layers=4,  # było 6
        memory_size=128,    # było 256
        smoothing_config_path=None,
        encoder_module=None,
        encoder_config_path=None
    ).to(device)

    print(f"Parametry modelu: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # --- optymalizator i funkcja straty ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.L1Loss()   # L1 (MAE) – do rekonstrukcji jasności

    best_val_loss = None

    for epoch in range(1, num_epochs + 1):
        # ===== TRAIN =====
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for gray_in, gray_target in train_loader:
            gray_in = gray_in.to(device)
            gray_target = gray_target.to(device)

            optimizer.zero_grad()

            # ColorMemoryTransformer zwraca [B, 1, H, W]
            output = model(gray_in)

            loss = criterion(output, gray_target)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # ===== VAL =====
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for gray_in, gray_target in val_loader:
                gray_in = gray_in.to(device)
                gray_target = gray_target.to(device)

                output = model(gray_in)
                loss = criterion(output, gray_target)

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches

        print(f"[Epoka {epoch}] train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f}")

        # zapis najlepszego modelu (wg val_loss)
        if best_val_loss is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(save_dir, "cmt_baseline_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> zapisano najlepszy model do {best_path}")

    # zapis końcowy
    final_path = os.path.join(save_dir, "cmt_baseline_last.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Zapisano model po ostatniej epoce do {final_path}")


if __name__ == "__main__":
    train_cmt()
