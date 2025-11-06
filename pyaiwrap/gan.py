import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, Any


def loadHyperparameters(json_path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing hyperparameters.

    Returns:
        Dict[str, Any]: A dictionary with hyperparameters and their values.
    """
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

    defaults = {
        "BATCH_SIZE": 1,
        "TRAIN_DATA_PATH": "./data/DIV2K_train_LR_bicubic/X4",
        "VALIDATION_DATA_PATH": "./data/DIV2K_valid_LR_bicubic/X4",
        "ARCHITECTURE_ID": "0",
        "HYPERPARAMS_ID": "0",
        "LEARNING_RATE": 0.0001,
        "GAMMA": 0.99,
        "IMAGE_RESIZE": 64,
        "IMAGE_CHANNELS": 3,
        "WARMUP_EPOCHS": 2,
        "EPOCHS": 100,
        "DIAGRAMS_DATA_PATH": "./diagrams_data",
        "WEIGHTS_PATH": "./weights",
        "PATIENCE": 15,
        "DIAGRAMS_PATH": "./diagrams",
        "VISUALIZE_EVERY": 10,
        "GRADIENT_CLIP": 1.0
    }

    for key, default_value in defaults.items():
        hyperparams.setdefault(key, default_value)

    return hyperparams


def warmupGAN(
    generator: nn.Module,
    discriminator: nn.Module,
    warmup_loader: torch.utils.data.DataLoader,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_epochs: int = 1,
    gradient_clip: float = 1.0
):
    """
    Warmup phase:
      1. Train generator as autoencoder (MSE reconstruction loss)
      2. Freeze generator
      3. Train discriminator on real/fake classification
      4. Unfreeze generator
    """

    mae_loss_fn = nn.L1Loss()
    generator.train()
    discriminator.train()

    # -------------------------
    # Generator Autoencoder Warmup
    # -------------------------
    for epoch in range(warmup_epochs):
        generator_losses = []
        generator_warmup_iterator = tqdm(warmup_loader, desc=f"Generator warmup epoch: {epoch+1}/{warmup_epochs}",
                                         position=0, leave=False)
        for target_images, _ in generator_warmup_iterator:
            target_images = target_images.to(device)

            generator_optimizer.zero_grad()

            reconstructed = generator(target_images)
            loss_g = mae_loss_fn(reconstructed, target_images)

            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=gradient_clip)

            loss_g.backward()
            generator_optimizer.step()
            generator_warmup_iterator.set_postfix(gen_loss=f"Generator warmup loss: {loss_g.item():.6f}")
            generator_losses.append(loss_g.item())
        print(f"Generator warmup epoch {epoch+1} mean loss: {np.mean(generator_losses):.6f}")

    for param in generator.parameters():
        param.requires_grad = False

    # -------------------------
    # Train Discriminator
    # -------------------------
    for epoch in range(warmup_epochs):
        discriminator_losses = []
        discriminator_warmup_iterator = tqdm(warmup_loader, desc=f"Discriminator warmup epoch: {epoch+1}/{warmup_epochs}",
                                             position=0, leave=False)
        for real_batch, _ in discriminator_warmup_iterator:
            target_images = real_batch.to(device)
            batch_size = target_images.size(0)

            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            discriminator_optimizer.zero_grad()

            outputs_real = discriminator(target_images)
            loss_real = mae_loss_fn(outputs_real, real_labels)

            with torch.no_grad():
                fake_images = generator(target_images)

            outputs_fake = discriminator(fake_images)
            loss_fake = mae_loss_fn(outputs_fake, fake_labels)

            loss_d = loss_real + loss_fake

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=gradient_clip)

            loss_d.backward()
            discriminator_optimizer.step()

            discriminator_warmup_iterator.set_postfix(discrim_loss=f"Discriminator warmup loss: {loss_d.item():.6f}")
            discriminator_losses.append(loss_d.item())
        print(f"Discriminator warmup epoch {epoch+1} mean loss: {np.mean(discriminator_losses):.6f}")

    for param in generator.parameters():
        param.requires_grad = True

    print("Warmup phase completed ✅")
