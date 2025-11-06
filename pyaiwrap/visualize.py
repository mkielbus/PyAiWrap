import torch
import os
from torchvision.utils import make_grid, save_image


def visualizeReconstruction(original_images: torch.Tensor,
                            modified_images: torch.Tensor,
                            reconstructed_images: torch.Tensor,
                            epoch: int,
                            save_path: str,
                            model_type: str,
                            launch_number: str,
                            hyperparams_id: str,
                            num_images: int = 8):
    """
    Create a visualization showing original, modified, and reconstructed images stacked vertically.
    Saves directly to file without displaying.
    Automatically converts single-channel images to 3-channel for visualization.

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
    """
    # Move to CPU and limit number of images
    original_images = original_images.detach().cpu()[:num_images]
    modified_images = modified_images.detach().cpu()[:num_images]
    reconstructed_images = reconstructed_images.detach().cpu()[:num_images]

    # Clamp values to valid range
    original_images = torch.clamp(original_images, 0, 1)
    modified_images = torch.clamp(modified_images, 0, 1)
    reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

    # Convert single-channel to 3-channel if needed
    if original_images.shape[1] == 1:
        original_images = original_images.repeat(1, 3, 1, 1)
    if modified_images.shape[1] == 1:
        modified_images = modified_images.repeat(1, 3, 1, 1)
    if reconstructed_images.shape[1] == 1:
        reconstructed_images = reconstructed_images.repeat(1, 3, 1, 1)

    # Stack vertically: original on top, modified middle, reconstructed bottom
    comparison = torch.cat([original_images, modified_images, reconstructed_images], dim=0)

    # Make grid with num_images per row
    grid = make_grid(comparison, nrow=num_images, padding=2, normalize=False)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path,
                             f'{model_type}_{launch_number}_{hyperparams_id}_reconstruction_epoch_{epoch:03d}.png')
    save_image(grid, save_file)
