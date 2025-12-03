import torch
import os
from torchvision.utils import make_grid, save_image
from typing import Optional
from .transforms import labToRgb, labToRgbForVisualization
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def convertToRgb(images: torch.Tensor,
                 channel_type: str,
                 paired_images: Optional[torch.Tensor] = None,
                 input_range: str = "zero_one") -> torch.Tensor:
    """
    Convert images to RGB based on channel type.

    Args:
        images: Tensor of shape (batch_size, C, H, W)
        channel_type: Type of channel ("RGB", "R", "G", "B", "LAB", "AB", "luminance")
        paired_images: For luminance+AB case, provide the paired channel
        input_range: "zero_one" for [0,1] or "kornia" for Kornia's native ranges

    Returns:
        RGB tensor in range [0, 1]
    """
    if channel_type == "RGB":
        return images

    elif channel_type == "luminance":
        if paired_images is not None and paired_images.shape[1] == 2:
            # Colorization: L + AB -> RGB
            if input_range == "zero_one":
                # Convert L from [0,1] to [0,100]
                l_channel = images * 100.0
                # AB should already be in Kornia's range from model output
                ab_channels = paired_images
            else:
                # Both already in Kornia ranges
                l_channel = images
                ab_channels = paired_images
            return labToRgb(l_channel, ab_channels)
        else:
            # Show L as grayscale
            if input_range == "zero_one":
                if images.shape[1] == 1:
                    return images.repeat(1, 3, 1, 1)
                else:
                    return images
            else:
                if images.shape[1] == 1:
                    return (images / 100.0).repeat(1, 3, 1, 1)
                else:
                    return images / 100.0

    elif channel_type == "AB":
        if paired_images is not None and paired_images.shape[1] == 1:
            # Colorization: L + AB -> RGB
            if input_range == "zero_one":
                # L from [0,1] to [0,100]
                l_channel = paired_images * 100.0
                # AB should be in Kornia range
                ab_channels = images * 254.0 - 127.0
            else:
                l_channel = paired_images
                ab_channels = images
            return labToRgb(l_channel, ab_channels)
        else:
            # Show AB as false color
            if input_range == "zero_one":
                # Convert AB from [0,1] to Kornia range for visualization
                ab_kornia = images * 254.0 - 127.0
            else:
                ab_kornia = images

            zeros = torch.zeros_like(ab_kornia[:, 0:1])
            lab_3ch = torch.cat([zeros, ab_kornia], dim=1)
            return labToRgbForVisualization(lab_3ch)

    elif channel_type == "LAB":
        if input_range == "zero_one":
            # Convert from [0,1] to Kornia ranges
            lab_kornia = torch.zeros_like(images)
            lab_kornia[:, 0:1] = images[:, 0:1] * 100.0  # L: [0,1] -> [0,100]
            lab_kornia[:, 1:3] = images[:, 1:3] * 254.0 - 127.0  # AB: [0,1] -> [-127,127]
            return labToRgbForVisualization(lab_kornia)
        else:
            return labToRgbForVisualization(images)

    elif channel_type == "R":
        zeros = torch.zeros_like(images)
        return torch.cat([images, zeros, zeros], dim=1)

    elif channel_type == "G":
        zeros = torch.zeros_like(images)
        return torch.cat([zeros, images, zeros], dim=1)

    elif channel_type == "B":
        zeros = torch.zeros_like(images)
        return torch.cat([zeros, zeros, images], dim=1)

    else:
        return images.repeat(1, 3, 1, 1)


def detect_range(images, channel_type):
    if channel_type in ["LAB", "AB"]:
        # Check if values are in [0,1] range (not Kornia range)
        if images.min() >= 0 and images.max() <= 1:
            return "zero_one"
        else:
            return "kornia"
    elif channel_type == "luminance":
        # Check if L is in [0,1] range (not [0,100] range)
        if images.min() >= 0 and images.max() <= 1:
            return "zero_one"
        else:
            return "kornia"
    else:
        return "zero_one"


def visualizeReconstruction(originalImages: torch.Tensor,
                            modifiedImages: torch.Tensor,
                            reconstructedImages: torch.Tensor,
                            epoch: int,
                            savePath: str,
                            modelType: str,
                            launchNumber: str,
                            hyperparamsId: str,
                            numImages: int = 8,
                            inputChannel: str = "RGB",
                            targetChannel: str = "RGB",
                            input_range: str = "auto") -> None:  # CHANGE TO "auto"
    """
    Create visualization for various input/output channel combinations.
    """
    originalImages = originalImages.detach().cpu()[:numImages]
    modifiedImages = modifiedImages.detach().cpu()[:numImages]
    reconstructedImages = reconstructedImages.detach().cpu()[:numImages]

    actualNumImages = min(numImages, originalImages.shape[0])

    if input_range == "auto":
        original_range = detect_range(originalImages, targetChannel)
        modified_range = detect_range(modifiedImages, inputChannel)
        reconstructed_range = detect_range(reconstructedImages, targetChannel)
    else:
        original_range = modified_range = reconstructed_range = input_range

    is_colorization = (inputChannel == "luminance" and targetChannel == "AB")
    is_ab_from_rgb = (inputChannel == "RGB" and targetChannel == "AB")
    is_lab_reconstruction = (inputChannel == "RGB" and targetChannel == "LAB")
    is_single_channel = (inputChannel in ["R", "G", "B", "luminance"] and targetChannel in ["R", "G", "B"])

    if is_colorization:
        # Colorization: L -> AB
        l_channel = modifiedImages * 100.0  # Convert L from [0,1] to [0,100]
        originalRgb = convertToRgb(originalImages, "AB", paired_images=l_channel, input_range=original_range)
        modifiedRgb = convertToRgb(modifiedImages, "luminance", input_range=modified_range)
        reconstructedRgb = convertToRgb(reconstructedImages, "AB", paired_images=l_channel, input_range=reconstructed_range)

    elif is_ab_from_rgb:
        l_channel = 0.299 * modifiedImages[:, 0:1] + 0.587 * modifiedImages[:, 1:2] + 0.114 * modifiedImages[:, 2:3]
        l_channel = l_channel * 100.0  # Convert to [0,100] range
        # Use the same L channel for both original and reconstructed AB
        originalRgb = convertToRgb(originalImages, "AB", paired_images=l_channel, input_range=original_range)
        modifiedRgb = convertToRgb(modifiedImages, "RGB", input_range=modified_range)
        reconstructedRgb = convertToRgb(reconstructedImages, "AB", paired_images=l_channel, input_range=reconstructed_range)

    elif is_lab_reconstruction:
        # LAB reconstruction: RGB -> LAB
        originalRgb = convertToRgb(originalImages, "LAB", input_range=original_range)
        modifiedRgb = convertToRgb(modifiedImages, "RGB", input_range=modified_range)
        reconstructedRgb = convertToRgb(reconstructedImages, "LAB", input_range=reconstructed_range)

    elif is_single_channel and inputChannel == targetChannel:
        # Single channel reconstruction
        originalRgb = convertToRgb(originalImages, targetChannel, input_range=original_range)
        modifiedRgb = convertToRgb(modifiedImages, inputChannel, input_range=modified_range)
        reconstructedRgb = convertToRgb(reconstructedImages, targetChannel, input_range=reconstructed_range)

    else:
        originalRgb = convertToRgb(originalImages, targetChannel, input_range=original_range)
        modifiedRgb = convertToRgb(modifiedImages, inputChannel, input_range=modified_range)
        reconstructedRgb = convertToRgb(reconstructedImages, targetChannel, input_range=reconstructed_range)

    originalRgb = torch.clamp(originalRgb, 0, 1)
    modifiedRgb = torch.clamp(modifiedRgb, 0, 1)
    reconstructedRgb = torch.clamp(reconstructedRgb, 0, 1)

    comparison = torch.cat([originalRgb, modifiedRgb, reconstructedRgb], dim=0)
    grid = make_grid(comparison, nrow=actualNumImages, padding=2, normalize=False)

    os.makedirs(savePath, exist_ok=True)
    saveFile = os.path.join(savePath,
                            f'{modelType}_{launchNumber}_{hyperparamsId}_epoch_{epoch:03d}_{inputChannel}_to_{targetChannel}.png')
    save_image(grid, saveFile)


def findMaskCenterSlice(mask_volume: np.ndarray) -> int:
    z_coords, y_coords, x_coords = np.where(mask_volume > 0)

    if len(z_coords) == 0:
        return mask_volume.shape[0] // 2

    centroid_z = int(np.mean(z_coords))
    return centroid_z


def visualizeSegmentation(
    volumes: torch.Tensor,
    true_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    epoch: int,
    save_path: str,
    phase: str,
    hyperparams_id: str,
    model_type: str,
    launch_number: int,
    class_colors: Dict[int, List[float]]
) -> None:
    os.makedirs(save_path, exist_ok=True)

    batch_size = min(4, volumes.shape[0])
    volumes_np = volumes.cpu().numpy()
    true_masks_np = true_masks.cpu().numpy()

    if true_masks_np.ndim == 5:
        true_masks_np = true_masks_np[:, 0]

    pred_masks_np = torch.softmax(pred_logits, dim=1).argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    if batch_size == 1:
        axes = axes.reshape(2, 1)

    for i in range(batch_size):
        slice_idx = findMaskCenterSlice(true_masks_np[i])

        input_slice = volumes_np[i, 0, slice_idx]
        true_mask_slice = true_masks_np[i, slice_idx]
        pred_mask_slice = pred_masks_np[i, slice_idx]

        true_colored = maskToColor(input_slice, true_mask_slice, class_colors)
        pred_colored = maskToColor(input_slice, pred_mask_slice, class_colors)

        axes[0, i].imshow(true_colored)
        axes[0, i].set_title(f'GT Sample {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(pred_colored)
        axes[1, i].set_title(f'Pred Sample {i+1}')
        axes[1, i].axis('off')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', label='Background'),
        plt.Rectangle((0, 0), 1, 1, facecolor=[0.2, 0.8, 0.2], label='Healthy'),
        plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.9, 0.2], label='Partially injured'),
        plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.3, 0.3], label='Completely ruptured')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(f'Epoch {epoch} - {phase.upper()} - Segmentation Results')
    plt.tight_layout()

    filename = f"{model_type}_{phase}_segmentation_epoch_{epoch}_hyperparams_{hyperparams_id}_{launch_number}.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()


def maskToColor(input_slice: np.ndarray, mask_slice: np.ndarray, class_colors: Dict[int, List[float]]) -> np.ndarray:
    input_normalized = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)

    colored = np.stack([input_normalized] * 3, axis=-1)

    for class_id, color in class_colors.items():
        mask = mask_slice == class_id
        if mask.any():
            for channel in range(3):
                colored_channel = colored[..., channel]
                colored_channel[mask] = color[channel] * 0.3 + colored_channel[mask] * 0.7
                colored[..., channel] = colored_channel

    return np.clip(colored, 0, 1)
