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

    Args:
        original_images: Tensor of shape (batch_size, C, H, W)
        modified_images: Tensor of shape (batch_size, C, H, W)
        reconstructed_images: Tensor of shape (batch_size, C, H, W)
        epoch: Current epoch number
        save_path: Directory to save the visualization
        num_images: Number of image triplets to show
    """
    original_images = original_images.detach().cpu()[:num_images]
    modified_images = modified_images.detach().cpu()[:num_images]
    reconstructed_images = reconstructed_images.detach().cpu()[:num_images]

    original_images = torch.clamp(original_images, 0, 1)
    modified_images = torch.clamp(modified_images, 0, 1)
    reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

    # Stack vertically: original on top, modified middle, reconstructed bottom
    comparison = torch.cat([original_images, modified_images, reconstructed_images], dim=0)

    # Make grid with num_images per row
    grid = make_grid(comparison, nrow=num_images, padding=2, normalize=False)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'{model_type}_{launch_number}_{hyperparams_id}_reconstruction_epoch_{epoch:03d}.png')
    save_image(grid, save_file)


def visualizeReconstructionGrid(modified_images, reconstructed_images, epoch, save_path, num_images=8):
    """
    Alternative visualization with side-by-side comparison.
    Saves directly to file without displaying.
    """
    modified_images = modified_images.detach().cpu()[:num_images]
    reconstructed_images = reconstructed_images.detach().cpu()[:num_images]

    modified_images = torch.clamp(modified_images, 0, 1)
    reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

    comparison = torch.stack([modified_images, reconstructed_images], dim=1)
    comparison = comparison.view(-1, *modified_images.shape[1:])

    grid = make_grid(comparison, nrow=2, padding=2, normalize=False)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'reconstruction_grid_epoch_{epoch:03d}.png')
    save_image(grid, save_file)
