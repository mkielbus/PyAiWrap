import torch
import os
from torchvision.utils import make_grid, save_image


def labToRgb(lChannel, abChannels):
    """
    Convert L and AB channels to RGB color space using pure PyTorch operations.

    Args:
        lChannel: L channel tensor of shape (batch_size, 1, H, W) in range [0, 100]
        abChannels: AB channels tensor of shape (batch_size, 2, H, W) in range [0, 255]

    Returns:
        RGB tensor in range [0, 1] with full gradient support
    """
    batchSize, _, H, W = lChannel.shape

    lab = torch.cat([lChannel, abChannels], dim=1)  # (batch_size, 3, H, W)

    L, A, B = lab[:, 0:1, :, :], lab[:, 1:2, :, :], lab[:, 2:3, :, :]

    L = L  # L in [0, 100]
    A = A - 128.0  # A was in [0, 255], convert back to [-128, 127]
    B = B - 128.0  # B was in [0, 255], convert back to [-128, 127]

    y = (L + 16.0) / 116.0
    x = (A / 500.0) + y
    z = y - (B / 200.0)

    xyz = torch.cat([x, y, z], dim=1)

    # Reverse nonlinearity
    mask = xyz > 0.2068966
    xyz = torch.where(mask, xyz ** 3, (xyz - 16.0/116.0) / 7.787)

    # Multiply by reference white (D65)
    reference_white = torch.tensor([0.95047, 1.0, 1.08883],
                                   device=xyz.device, dtype=xyz.dtype).view(1, 3, 1, 1)
    xyz = xyz * reference_white

    xyz_to_rgb = torch.tensor([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], device=xyz.device, dtype=xyz.dtype)

    # (B, 3, H, W) -> (B, H, W, 3)
    xyz_permuted = xyz.permute(0, 2, 3, 1)
    rgb_permuted = torch.matmul(xyz_permuted, xyz_to_rgb.t())
    rgb = rgb_permuted.permute(0, 3, 1, 2)

    # Apply gamma correction
    mask = rgb > 0.0031308
    rgb = torch.where(mask, 1.055 * (rgb ** (1/2.4)) - 0.055, 12.92 * rgb)

    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb


def _labToRgbSingle(lab):
    """
    Convert single LAB image to RGB.

    Args:
        lab: LAB image of shape (H, W, 3)

    Returns:
        RGB image of shape (H, W, 3) in range [0, 1]
    """
    import numpy as np

    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    L = L  # L in [0, 100]
    A = A - 128  # A in [0, 255], convert to [-128, 127]
    B = B - 128  # B in [0, 255], convert to [-128, 127]

    y = (L + 16) / 116
    x = (A / 500) + y
    z = y - (B / 200)

    xyz = np.stack([x, y, z], axis=-1)

    mask = xyz > 0.2068966
    xyz[mask] = xyz[mask] ** 3
    xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

    xyz = xyz * np.array([0.95047, 1.0, 1.08883])

    xyzToRgb = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])

    rgb = np.dot(xyz, xyzToRgb.T)

    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1/2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]

    rgb = np.clip(rgb, 0, 1)
    return rgb.astype(np.float32)


def _labToRgbTensorNumpy(labTensor):
    """
    Convert LAB tensor to RGB tensor using numpy (for visualization only).

    Args:
        labTensor: LAB tensor of shape (batch_size, 3, H, W)

    Returns:
        RGB tensor in range [0, 1]
    """
    batchSize, _, H, W = labTensor.shape
    labNp = labTensor.permute(0, 2, 3, 1).cpu().numpy()

    rgbBatch = []
    for i in range(batchSize):
        rgbImg = _labToRgbSingle(labNp[i])
        rgbBatch.append(torch.from_numpy(rgbImg).permute(2, 0, 1))

    return torch.stack(rgbBatch)


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
            # AB as false color (approximation)
            ab3ch = torch.cat([
                torch.zeros_like(images[:, 0:1]),  # Zero L channel
                images  # AB channels
            ], dim=1)
            return _labToRgbTensorNumpy(ab3ch)

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
