import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pyaiwrap.utils import prepareDevice
from pyaiwrap.datasets import SimpleColorizationDataset
from pyaiwrap.neural_network import ColorizationTransformerNet


class SimpleFusionNet(nn.Module):
    """
    Mały konwolucyjny „refiner”:
    wejście: [B,3,H,W] (RGB z 3 CTN-ów)
    wyjście: [B,3,H,W] (ulepszone RGB)
    """
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )

    def forward(self, x):
        # residual: wyjście = x + poprawka
        correction = self.net(x)
        return torch.clamp(x + correction, 0.0, 1.0)


def load_single_ctn(weights_path: str, device: torch.device) -> nn.Module:
    """
    Ładuje jeden CTN (output_channels=1) z wagami.
    """
    image_size = 128
    model = ColorizationTransformerNet(
        embed_dim=128,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.1,
        num_layers=2,
        num_color_tokens=1024,
        num_image_patches=1024,
        image_size=image_size,
        use_decoder_masking=False,
        only_use_encoder=False,
        output_channels=1,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def train_fusion():
    # ------------------ USTAWIENIA ------------------
    image_size = 128
    batch_size = 2
    num_epochs = 3
    learning_rate = 2e-4
    num_workers = 0 if os.name == "nt" else 4

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    device = prepareDevice(use_cuda=True)

    # ------------------ CTN-y (zamrożone) ------------------
    ctn_r_path = os.path.join(weights_dir, "ctn_r_best.pth")
    ctn_g_path = os.path.join(weights_dir, "ctn_g_best.pth")
    ctn_b_path = os.path.join(weights_dir, "ctn_b_best.pth")

    assert os.path.exists(ctn_r_path), f"Brak {ctn_r_path} – najpierw wytrenuj CTN dla kanału R"
    assert os.path.exists(ctn_g_path), f"Brak {ctn_g_path} – najpierw wytrenuj CTN dla kanału G"
    assert os.path.exists(ctn_b_path), f"Brak {ctn_b_path} – najpierw wytrenuj CTN dla kanału B"

    ctn_r = load_single_ctn(ctn_r_path, device)
    ctn_g = load_single_ctn(ctn_g_path, device)
    ctn_b = load_single_ctn(ctn_b_path, device)

    # ------------------ DANE ------------------
    train_dataset = SimpleColorizationDataset(
        root_dir="data/train",
        image_size=image_size
    )
    val_dataset = SimpleColorizationDataset(
        root_dir="data/val",
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ------------------ FUSION NET ------------------
    fusion_net = SimpleFusionNet(in_channels=3, hidden_channels=32).to(device)
    num_params = sum(p.numel() for p in fusion_net.parameters())
    print(f"Parametry FusionNet: {num_params/1e6:.2f} M")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(fusion_net.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_path = os.path.join(weights_dir, "fusion_best.pth")
    last_path = os.path.join(weights_dir, "fusion_last.pth")

    # ------------------ PĘTLA TRENINGOWA ------------------
    for epoch in range(1, num_epochs + 1):
        fusion_net.train()
        running_train_loss = 0.0

        for gray, color in train_loader:
            gray = gray.to(device)   # [B,1,H,W]
            color = color.to(device) # [B,3,H,W]

            with torch.no_grad():
                r = ctn_r(gray)  # [B,1,H,W]
                g = ctn_g(gray)
                b = ctn_b(gray)

            rgb_init = torch.cat([r, g, b], dim=1)  # [B,3,H,W]

            optimizer.zero_grad()
            refined = fusion_net(rgb_init)
            loss = criterion(refined, color)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * gray.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # --------- WALIDACJA ---------
        fusion_net.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for gray, color in val_loader:
                gray = gray.to(device)
                color = color.to(device)

                r = ctn_r(gray)
                g = ctn_g(gray)
                b = ctn_b(gray)
                rgb_init = torch.cat([r, g, b], dim=1)

                refined = fusion_net(rgb_init)
                loss = criterion(refined, color)
                running_val_loss += loss.item() * gray.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)

        print(
            f"[Fusion Epoka {epoch}] "
            f"train_loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(fusion_net.state_dict(), best_path)
            print(f"  -> zapisano najlepszy FusionNet do {best_path}")

    torch.save(fusion_net.state_dict(), last_path)
    print(f"Zapisano FusionNet po ostatniej epoce do {last_path}")


if __name__ == "__main__":
    train_fusion()
