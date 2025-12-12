from typing import Dict, Any, List, Tuple
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from pyaiwrap.utils import prepareDevice
from pyaiwrap.neural_network import ColorizationTransformerNet
from pyaiwrap.datasets import SimpleColorizationDataset
from pyaiwrap.train import train



class SimpleMetrics:
    def __init__(self):
        self.current_phase = "train"
        self._batch_losses: List[float] = []
        self.history: Dict[Tuple[int, str], Dict[str, float]] = {}

    def setPhase(self, phase: str):
        self.current_phase = phase
        self._batch_losses = []

    def addBatchLoss(self, value: float):
        self._batch_losses.append(value)

    def finalizeEpoch(self, epoch: int):
        if self._batch_losses:
            avg_loss = sum(self._batch_losses) / len(self._batch_losses)
        else:
            avg_loss = 0.0
        self.history[(epoch, self.current_phase)] = {"loss": avg_loss}

    def display(self, epoch: int):
        train_loss = self.history.get((epoch, "train"), {}).get("loss", float("nan"))
        val_loss = self.history.get((epoch, "val"), {}).get("loss", float("nan"))
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    def save(self, diagrams_data_path: str, hyperparams_id: str, model_type: str, launch_number: int):
        os.makedirs(diagrams_data_path, exist_ok=True)

    def getMetric(self, epoch: int, phase: str, name: str) -> float:
        return self.history.get((epoch, phase), {}).get(name, float("inf"))


def main():
    device = prepareDevice(use_cuda=True)

    image_size = 256

    # --- Datasets ---
    train_dataset = SimpleColorizationDataset(root_dir="data/train", image_size=image_size)
    val_dataset   = SimpleColorizationDataset(root_dir="data/val",   image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False,
                              num_workers=0, pin_memory=True)

    # --- Model ---
    model = ColorizationTransformerNet(
        embed_dim=256,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        num_layers=4,
        num_color_tokens=64,     # 8x8
        num_image_patches=64,    # 8x8
        image_size=image_size,
        use_decoder_masking=False,
        only_use_encoder=False,
        output_channels=3,
    ).to(device)

    models = {"generator": model}

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizers = {"generator": optimizer}

    # --- Metrics ---
    metrics = SimpleMetrics()

    # --- Loss (MAE) ---
    l1_loss = nn.L1Loss()

    def loss_fn(models_dict: Dict[str, nn.Module],
                batch,
                metrics_obj: SimpleMetrics,
                gradient_clip: float | None) -> Dict[str, Any]:
        model = models_dict["generator"]
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        preds = model(inputs)

        loss = l1_loss(preds, targets)

        metrics_obj.addBatchLoss(loss.item())

        if gradient_clip is not None:
            loss.backward()
            if gradient_clip > 0:
                clip_grad_norm_(model.parameters(), gradient_clip)

        return {"loss": loss}

    results = train(
        models=models,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizers=optimizers,
        loss_fn=loss_fn,
        metrics=metrics,
        schedulers=None,
        device=device,
        num_epochs=50,                       # na próbę
        diagrams_data_path="./diagrams_data",
        hyperparams_id="first_run",
        weights_path="./weights",
        diagrams_path="./diagrams",
        launch_number=0,
        visualize_every_xth_epoch=None,
        max_patience=1000,
        model_type="color_transformer",
        gradient_clip=1.0,
        control_fn=None,
        early_stopping_metric="loss",
    )

    print("Trening zakończony.")
    print("Wyniki:", results)


if __name__ == "__main__":
    main()
