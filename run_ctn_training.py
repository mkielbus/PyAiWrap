import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pyaiwrap.utils import prepareDevice
from pyaiwrap.datasets import SimpleColorizationDataset
from pyaiwrap.neural_network import ColorizationTransformerNet


def train_single_channel(channel: str = "r"):
    """
    Trenuje ColorizationTransformerNet do pojedynczego kanału:
    - 'r' -> kanał 0
    - 'g' -> kanał 1
    - 'b' -> kanał 2
    """
    assert channel in ["r", "g", "b"], "channel musi być jednym z: r, g, b"
    channel_index = {"r": 0, "g": 1, "b": 2}[channel]

    # ------------------ USTAWIENIA ------------------
    image_size = 128
    batch_size = 2
    num_epochs = 3
    learning_rate = 2e-4
    num_workers = 0 if os.name == "nt" else 4

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    device = prepareDevice(use_cuda=True)
    print(f"Trenuję kanał: {channel.upper()} (indeks {channel_index})")

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

    # ------------------ MODEL ------------------
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
        output_channels=1,   # <---- JEDEN KANAŁ
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parametry CTN ({channel.upper()}): {num_params/1e6:.2f} M")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_path = os.path.join(weights_dir, f"ctn_{channel}_best.pth")
    last_path = os.path.join(weights_dir, f"ctn_{channel}_last.pth")

    # ------------------ PĘTLA TRENINGOWA ------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for gray, color in train_loader:
            gray = gray.to(device)             # [B, 1, H, W]
            color = color.to(device)           # [B, 3, H, W]
            target = color[:, channel_index:channel_index+1, :, :]  # [B, 1, H, W]

            optimizer.zero_grad()
            output = model(gray)               # [B, 1, H, W]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * gray.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # ---------- WALIDACJA ----------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for gray, color in val_loader:
                gray = gray.to(device)
                color = color.to(device)
                target = color[:, channel_index:channel_index+1, :, :]

                output = model(gray)
                loss = criterion(output, target)
                running_val_loss += loss.item() * gray.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)

        print(
            f"[CTN {channel.upper()} Epoka {epoch}] "
            f"train_loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f}"
        )

        # zapis najlepszego
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> zapisano najlepszy CTN_{channel.upper()} do {best_path}")

    torch.save(model.state_dict(), last_path)
    print(f"Zapisano CTN_{channel.upper()} po ostatniej epoce do {last_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel",
        type=str,
        default="r",
        help="Kanał do trenowania: r, g lub b"
    )
    args = parser.parse_args()

    train_single_channel(args.channel.lower())
