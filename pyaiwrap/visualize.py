import torch
import os
from torchvision.utils import make_grid, save_image
from .transforms import labToRgb, labToRgbForVisualization


def convertToRgb(images, channelType, pairedImages=None):
    """
    Convert images to RGB based on channel type.

    Args:
        images: Tensor of shape (batch_size, C, H, W)
        channelType: Type of channel ("luminance", "R", "G", "B", "RGB", "ab")
        pairedImages: For luminance+ab case, provide the paired channel

    Returns:
        RGB tensor in range [0, 1]
    """
    if channelType == "RGB":
        return images

    elif channelType == "luminance":
        if pairedImages is not None and pairedImages.shape[1] == 2:
            # L from [0,1] to [0,100] range
            lChannel = images * 100.0
            return labToRgb(lChannel, pairedImages)
        else:
            return images.repeat(1, 3, 1, 1)

    elif channelType == "ab":
        if pairedImages is not None and pairedImages.shape[1] == 1:
            # L from [0,1] to [0,100] range
            lChannel = pairedImages * 100.0
            return labToRgb(lChannel, images)
        else:
            # AB as false color (approximation) - use PyTorch conversion
            ab3ch = torch.cat([
                torch.zeros_like(images[:, 0:1]),  # Zero L channel
                images  # AB channels
            ], dim=1)
            return labToRgbForVisualization(ab3ch * 255.0 - 128.0)

    elif channelType == "R":
        return torch.cat([images, torch.zeros_like(images), torch.zeros_like(images)], dim=1)

    elif channelType == "G":
        return torch.cat([torch.zeros_like(images), images, torch.zeros_like(images)], dim=1)

    elif channelType == "B":
        return torch.cat([torch.zeros_like(images), torch.zeros_like(images), images], dim=1)

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
                            targetChannel: str = "RGB",
                            inputChannel: str = "RGB"):
    """
    Create a visualization showing original, modified, and reconstructed images stacked vertically.
    Handles luminance+AB to RGB conversion for colorization tasks.

    Args:
        originalImages: Tensor of shape (batch_size, C, H, W)
        modifiedImages: Tensor of shape (batch_size, C, H, W)
        reconstructedImages: Tensor of shape (batch_size, C, H, W)
        epoch: Current epoch number
        savePath: Directory to save the visualization
        modelType: Type of model (for filename)
        launchNumber: Launch number (for filename)
        hyperparamsId: Hyperparameters ID (for filename)
        numImages: Number of image triplets to show
        targetChannel: Target channel type ("luminance", "R", "G", "B", "RGB", "ab")
        inputChannel: Input channel type ("luminance", "R", "G", "B", "RGB", "ab")
    """
    originalImages = originalImages.detach().cpu()[:numImages]
    modifiedImages = modifiedImages.detach().cpu()[:numImages]
    reconstructedImages = reconstructedImages.detach().cpu()[:numImages]

    actualNumImages = min(numImages, originalImages.shape[0])

    originalImages = torch.clamp(originalImages, 0, 1)
    modifiedImages = torch.clamp(modifiedImages, 0, 1)
    reconstructedImages = torch.clamp(reconstructedImages, 0, 1)

    isLab = (inputChannel == "luminance" and targetChannel == "ab")

    if isLab:
        originalRgb = convertToRgb(originalImages, "RGB")
        modifiedRgb = convertToRgb(modifiedImages, "luminance")  # Show as grayscale
        reconstructedRgb = convertToRgb(reconstructedImages, "ab", modifiedImages)
    else:
        originalRgb = convertToRgb(originalImages, targetChannel)
        modifiedRgb = convertToRgb(modifiedImages, inputChannel)
        reconstructedRgb = convertToRgb(reconstructedImages, targetChannel)

    # Stack vertically: original, modified, reconstructed
    comparison = torch.cat([originalRgb, modifiedRgb, reconstructedRgb], dim=0)

    grid = make_grid(comparison, nrow=actualNumImages, padding=2, normalize=False)

    os.makedirs(savePath, exist_ok=True)
    saveFile = os.path.join(savePath,
                            f'{modelType}_{launchNumber}_{hyperparamsId}_reconstruction_epoch_{epoch:03d}.png')
    save_image(grid, saveFile)
