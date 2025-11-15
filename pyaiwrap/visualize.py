import torch
import os
from torchvision.utils import make_grid, save_image
from typing import Optional
from .transforms import labToRgb, labToRgbForVisualization


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
                return images.repeat(1, 3, 1, 1)
            else:
                return (images / 100.0).repeat(1, 3, 1, 1)

    elif channel_type == "AB":
        if paired_images is not None and paired_images.shape[1] == 1:
            # Colorization: L + AB -> RGB
            if input_range == "zero_one":
                # L from [0,1] to [0,100]
                l_channel = paired_images * 100.0
                # AB should be in Kornia range
                ab_channels = images
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
                            input_range: str = "zero_one") -> None:
    """
    Create visualization for various input/output channel combinations.
    """
    originalImages = originalImages.detach().cpu()[:numImages]
    modifiedImages = modifiedImages.detach().cpu()[:numImages]
    reconstructedImages = reconstructedImages.detach().cpu()[:numImages]

    actualNumImages = min(numImages, originalImages.shape[0])

    is_colorization = (inputChannel == "luminance" and targetChannel == "AB")
    is_ab_from_rgb = (inputChannel == "RGB" and targetChannel == "AB")
    is_lab_reconstruction = (inputChannel == "RGB" and targetChannel == "LAB")
    is_single_channel = (inputChannel in ["R", "G", "B"] and targetChannel in ["R", "G", "B"])

    if is_colorization:
        # Colorization: L -> AB
        # Original: RGB, Modified: L, Reconstructed: AB
        originalRgb = convertToRgb(originalImages, "RGB", input_range=input_range)
        modifiedRgb = convertToRgb(modifiedImages, "luminance", input_range=input_range)
        reconstructedRgb = convertToRgb(reconstructedImages, "AB", modifiedImages, input_range=input_range)

    elif is_ab_from_rgb:
        # AB prediction from RGB
        # Original: AB, Modified: RGB, Reconstructed: AB
        originalRgb = convertToRgb(originalImages, "AB", input_range=input_range)
        modifiedRgb = convertToRgb(modifiedImages, "RGB", input_range=input_range)
        reconstructedRgb = convertToRgb(reconstructedImages, "AB", input_range=input_range)

    elif is_lab_reconstruction:
        # LAB reconstruction: RGB -> LAB
        # Original: LAB, Modified: RGB, Reconstructed: LAB
        originalRgb = convertToRgb(originalImages, "LAB", input_range=input_range)
        modifiedRgb = convertToRgb(modifiedImages, "RGB", input_range=input_range)
        reconstructedRgb = convertToRgb(reconstructedImages, "LAB", input_range=input_range)

    elif is_single_channel and inputChannel == targetChannel:
        # Single channel reconstruction (R->R, G->G, B->B)
        originalRgb = convertToRgb(originalImages, targetChannel, input_range=input_range)
        modifiedRgb = convertToRgb(modifiedImages, inputChannel, input_range=input_range)
        reconstructedRgb = convertToRgb(reconstructedImages, targetChannel, input_range=input_range)

    else:
        # Generic case - handle all other combinations
        originalRgb = convertToRgb(originalImages, targetChannel, input_range=input_range)
        modifiedRgb = convertToRgb(modifiedImages, inputChannel, input_range=input_range)
        reconstructedRgb = convertToRgb(reconstructedImages, targetChannel, input_range=input_range)

    # NOW clamp the RGB images to [0,1] for saving
    originalRgb = torch.clamp(originalRgb, 0, 1)
    modifiedRgb = torch.clamp(modifiedRgb, 0, 1)
    reconstructedRgb = torch.clamp(reconstructedRgb, 0, 1)

    # Stack vertically: original, modified, reconstructed
    comparison = torch.cat([originalRgb, modifiedRgb, reconstructedRgb], dim=0)

    grid = make_grid(comparison, nrow=actualNumImages, padding=2, normalize=False)

    os.makedirs(savePath, exist_ok=True)
    saveFile = os.path.join(savePath,
                            f'{modelType}_{launchNumber}_{hyperparamsId}_epoch_{epoch:03d}_{inputChannel}_to_{targetChannel}.png')
    save_image(grid, saveFile)
    print(f"Saved visualization: {inputChannel} -> {targetChannel} to {saveFile}")
