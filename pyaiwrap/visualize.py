import torch
import os
from torchvision.utils import make_grid, save_image


def convert_single_channel(images, channel_type):
    if images.shape[1] == 3:
        return images

    if channel_type == "R":
        # [R, 0, 0]
        return torch.cat([
            images,  # R
            torch.zeros_like(images),  # G
            torch.zeros_like(images)   # B
        ], dim=1)
    elif channel_type == "G":
        # [0, G, 0]
        return torch.cat([
            torch.zeros_like(images),  # R
            images,  # G
            torch.zeros_like(images)   # B
        ], dim=1)
    elif channel_type == "B":
        # [0, 0, B]
        return torch.cat([
            torch.zeros_like(images),  # R
            torch.zeros_like(images),  # G
            images   # B
        ], dim=1)
    else:  # grayscale (all channels equal)
        return images.repeat(1, 3, 1, 1)


def visualizeReconstruction(original_images: torch.Tensor,
                            modified_images: torch.Tensor,
                            reconstructed_images: torch.Tensor,
                            epoch: int,
                            save_path: str,
                            model_type: str,
                            launch_number: str,
                            hyperparams_id: str,
                            num_images: int = 8,
                            target_channel: str = "RGB"):
    """
    Create a visualization showing original, modified, and reconstructed images stacked vertically.
    For single-channel images, converts to 3-channel with appropriate zeroing.

    Args:
        original_images: Tensor of shape (batch_size, C, H, W) where C can be 1 or 3
        modified_images: Tensor of shape (batch_size, C, H, W) where C can be 1 or 3
        reconstructed_images: Tensor of shape (batch_size, C, H, W) where C can be 1 or 3
        epoch: Current epoch number
        save_path: Directory to save the visualization
        model_type: Type of model (for filename)
        launch_number: Launch number (for filename)
        hyperparams_id: Hyperparameters ID (for filename)
        num_images: Number of image triplets to show
        target_channel: Target channel for single-channel models ("R", "G", "B", or "RGB")
    """
    original_images = original_images.detach().cpu()[:num_images]
    modified_images = modified_images.detach().cpu()[:num_images]
    reconstructed_images = reconstructed_images.detach().cpu()[:num_images]

    actual_num_images = min(num_images, original_images.shape[0])

    original_images = torch.clamp(original_images, 0, 1)
    modified_images = torch.clamp(modified_images, 0, 1)
    reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

    original_3ch = convert_single_channel(original_images, target_channel)
    modified_3ch = convert_single_channel(modified_images, "RGB")
    reconstructed_3ch = convert_single_channel(reconstructed_images, target_channel)

    # Stack vertically: top, modified, reconstructed
    comparison = torch.cat([original_3ch, modified_3ch, reconstructed_3ch], dim=0)

    grid = make_grid(comparison, nrow=actual_num_images, padding=2, normalize=False)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path,
                             f'{model_type}_{launch_number}_{hyperparams_id}_reconstruction_epoch_{epoch:03d}.png')
    save_image(grid, save_file)
