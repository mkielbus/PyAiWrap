import torch
import os
from torchvision.utils import make_grid, save_image
from typing import Optional, Dict, List, Tuple
from .transforms import labToRgb, labToRgbForVisualization
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class VisualizationStrategy(ABC):
    """Abstract base class for visualization strategies."""

    @abstractmethod
    def visualize(self, *args, **kwargs) -> None:
        pass


class ImageConverter:
    """Handles image conversion between different color spaces."""

    def __init__(self):
        self.channel_handlers = {
            "RGB": self.handleRgb,
            "luminance": self.handleLuminance,
            "AB": self.handleAb,
            "LAB": self.handleLab,
            "R": self.handleSingleChannel,
            "G": self.handleSingleChannel,
            "B": self.handleSingleChannel,
        }

    def convert(self, images: torch.Tensor, channel_type: str,
                paired_images: Optional[torch.Tensor] = None,
                input_range: str = "zero_one") -> torch.Tensor:
        """Convert images to RGB based on channel type."""
        handler = self.channel_handlers.get(channel_type, self.handleDefault)
        return handler(images, channel_type, paired_images, input_range)

    def handleRgb(self, images: torch.Tensor, channel_type: str,
                  paired_images: Optional[torch.Tensor], input_range: str) -> torch.Tensor:
        return images

    def handleLuminance(self, images: torch.Tensor, channel_type: str,
                        paired_images: Optional[torch.Tensor], input_range: str) -> torch.Tensor:
        if self.isLToABCase(images, paired_images):
            return self.convertLtoRgb(images, paired_images, input_range)
        return self.convertLtoGrayscale(images, input_range)

    def handleAb(self, images: torch.Tensor, channel_type: str,
                 paired_images: Optional[torch.Tensor], input_range: str) -> torch.Tensor:
        if self.isLToABCase(images, paired_images):
            return self.convertAbToRgb(images, paired_images, input_range)
        return self.convertAbToFalseColor(images, input_range)

    def handleLab(self, images: torch.Tensor, channel_type: str,
                  paired_images: Optional[torch.Tensor], input_range: str) -> torch.Tensor:
        if input_range == "zero_one":
            images = self.normalizeLabToKornia(images)
        return labToRgbForVisualization(images)

    def handleSingleChannel(self, images: torch.Tensor, channel_type: str,
                            paired_images: Optional[torch.Tensor], input_range: str) -> torch.Tensor:
        channelMap = {"R": 0, "G": 1, "B": 2}
        zeros = torch.zeros_like(images)

        if channel_type in channelMap:
            rgbChannels = [zeros, zeros, zeros]
            rgbChannels[channelMap[channel_type]] = images
            return torch.cat(rgbChannels, dim=1)
        return images.repeat(1, 3, 1, 1)

    def handleDefault(self, images: torch.Tensor, channel_type: str,
                      paired_images: Optional[torch.Tensor], input_range: str) -> torch.Tensor:
        return images.repeat(1, 3, 1, 1)

    def isLToABCase(self, images: torch.Tensor,
                    paired_images: Optional[torch.Tensor]) -> bool:
        return (paired_images is not None and
                ((images.shape[1] == 1 and paired_images.shape[1] == 2) or
                 (images.shape[1] == 2 and paired_images.shape[1] == 1)))

    def convertLtoRgb(self, l_channel: torch.Tensor, ab_channels: torch.Tensor,
                      inputRange: str) -> torch.Tensor:
        if inputRange == "zero_one":
            l_channel = l_channel * 100.0
        return labToRgb(l_channel, ab_channels)

    def convertLtoGrayscale(self, images: torch.Tensor, input_range: str) -> torch.Tensor:
        if images.shape[1] == 1:
            if input_range == "zero_one":
                return images.repeat(1, 3, 1, 1)
            return (images / 100.0).repeat(1, 3, 1, 1)
        return images / 100.0 if input_range != "zero_one" else images

    def convertAbToRgb(self, images: torch.Tensor, l_channel: torch.Tensor,
                       input_range: str) -> torch.Tensor:
        if input_range == "zero_one":
            l_channel = l_channel * 100.0
            ab_channels = images * 254.0 - 127.0
        else:
            ab_channels = images
        return labToRgb(l_channel, ab_channels)

    def convertAbToFalseColor(self, images: torch.Tensor, input_range: str) -> torch.Tensor:
        if input_range == "zero_one":
            images = images * 254.0 - 127.0

        zeros = torch.zeros_like(images[:, 0:1])
        lab3ch = torch.cat([zeros, images], dim=1)
        return labToRgbForVisualization(lab3ch)

    def normalizeLabToKornia(self, images: torch.Tensor) -> torch.Tensor:
        lab_kornia = torch.zeros_like(images)
        lab_kornia[:, 0:1] = images[:, 0:1] * 100.0
        lab_kornia[:, 1:3] = images[:, 1:3] * 254.0 - 127.0
        return lab_kornia


class RangeDetector:
    """Detects input range based on channel type and values."""

    @staticmethod
    def detect(images: torch.Tensor, channel_type: str) -> str:
        if channel_type in ["LAB", "AB"]:
            return RangeDetector.detectLabRange(images)
        elif channel_type == "luminance":
            return RangeDetector.detectLuminanceRange(images)
        return "zero_one"

    @staticmethod
    def detectLabRange(images: torch.Tensor) -> str:
        return "zero_one" if images.min() >= 0 and images.max() <= 1 else "kornia"

    @staticmethod
    def detectLuminanceRange(images: torch.Tensor) -> str:
        return "zero_one" if images.min() >= 0 and images.max() <= 1 else "kornia"


class ColorizationVisualizer(VisualizationStrategy):
    """Handles visualization of colorization/reconstruction tasks."""

    def __init__(self):
        self.converter = ImageConverter()
        self.range_detector = RangeDetector()

    def visualize(self, original_images: torch.Tensor,
                  modified_images: torch.Tensor,
                  reconstructed_images: torch.Tensor,
                  epoch: int,
                  save_path: str,
                  model_type: str,
                  launch_number: str,
                  config_id: str,
                  num_images: int = 8,
                  input_channel: str = "RGB",
                  target_channel: str = "RGB",
                  input_range: str = "auto") -> None:

        original = original_images.detach().cpu()[:num_images]
        modified = modified_images.detach().cpu()[:num_images]
        reconstructed = reconstructed_images.detach().cpu()[:num_images]

        ranges = self.getRanges(original, modified, reconstructed,
                                input_channel, target_channel, input_range)

        original_rgb, modified_rgb, reconstructed_rgb = self.convertToRgb(
            original, modified, reconstructed, input_channel, target_channel, ranges)

        self.saveGrid(original_rgb, modified_rgb, reconstructed_rgb,
                      epoch, save_path, model_type, launch_number,
                      config_id, input_channel, target_channel)

    def getRanges(self, original: torch.Tensor, modified: torch.Tensor,
                  reconstructed: torch.Tensor, input_channel: str,
                  target_channel: str, input_range: str) -> Tuple[str, str, str]:
        if input_range == "auto":
            return (
                self.range_detector.detect(original, target_channel),
                self.range_detector.detect(modified, input_channel),
                self.range_detector.detect(reconstructed, target_channel)
            )
        return input_range, input_range, input_range

    def convertToRgb(self, original: torch.Tensor, modified: torch.Tensor,
                     reconstructed: torch.Tensor, input_channel: str,
                     target_channel: str, ranges: Tuple[str, str, str]) -> Tuple[torch.Tensor, ...]:

        original_range, modified_range, reconstructed_range = ranges

        if self.isColorizationCase(input_channel, target_channel):
            return self.handleColorizationCase(original, modified, reconstructed,
                                               original_range, modified_range, reconstructed_range)

        original_rgb = self.converter.convert(original, target_channel,
                                             input_range=original_range)
        modified_rgb = self.converter.convert(modified, input_channel,
                                             input_range=modified_range)
        reconstructed_rgb = self.converter.convert(reconstructed, target_channel,
                                                  input_range=reconstructed_range)

        return self.clampImages(original_rgb, modified_rgb, reconstructed_rgb)

    def isColorizationCase(self, input_channel: str, target_channel: str) -> bool:
        colorizationCases = [
            (input_channel == "luminance" and target_channel == "AB"),
            (input_channel == "RGB" and target_channel == "AB"),
            (input_channel == "RGB" and target_channel == "LAB"),
            (input_channel in ["R", "G", "B", "luminance"] and
             target_channel in ["R", "G", "B"] and input_channel == target_channel)
        ]
        return any(colorizationCases)

    def handleColorizationCase(self, original: torch.Tensor, modified: torch.Tensor,
                               reconstructed: torch.Tensor, original_range: str,
                               modified_range: str, reconstructed_range: str) -> Tuple[torch.Tensor, ...]:

        if modified.shape[1] == 3:  # RGB input
            l_channel = self.calculateLuminance(modified) * 100.0
        else:
            l_channel = modified * 100.0 if modified_range == "zero_one" else modified

        original_rgb = self.converter.convert(original, "AB",
                                              paired_images=l_channel,
                                              input_range=original_range)
        modified_rgb = self.converter.convert(modified, "luminance" if modified.shape[1] == 1 else "RGB",
                                              input_range=modified_range)
        reconstructed_rgb = self.converter.convert(reconstructed, "AB",
                                                   paired_images=l_channel,
                                                   input_range=reconstructed_range)

        return self.clampImages(original_rgb, modified_rgb, reconstructed_rgb)

    def calculateLuminance(self, rgb_images: torch.Tensor) -> torch.Tensor:
        return 0.299 * rgb_images[:, 0:1] + 0.587 * rgb_images[:, 1:2] + 0.114 * rgb_images[:, 2:3]

    def clampImages(self, *images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.clamp(img, 0, 1) for img in images)

    def saveGrid(self, original_rgb: torch.Tensor, modified_rgb: torch.Tensor,
                 reconstructed_rgb: torch.Tensor, epoch: int, save_path: str,
                 model_type: str, launch_number: str, config_id: str,
                 inputChannel: str, targetChannel: str) -> None:

        os.makedirs(save_path, exist_ok=True)

        comparison = torch.cat([original_rgb, modified_rgb, reconstructed_rgb], dim=0)
        grid = make_grid(comparison, nrow=original_rgb.shape[0], padding=2, normalize=False)

        saveFile = os.path.join(
            save_path,
            f'{model_type}_{launch_number}_{config_id}_epoch_{epoch:03d}_{inputChannel}_to_{targetChannel}.png'
        )
        save_image(grid, saveFile)


class MaskCenterFinder:
    """Finds the center slice of a mask volume."""

    @staticmethod
    def find(mask_volume: np.ndarray) -> int:
        z_coords, y_coords, x_coords = np.where(mask_volume > 0)
        return int(np.mean(z_coords)) if len(z_coords) > 0 else mask_volume.shape[0] // 2


class MaskColorizer:
    """Handles colorization of mask overlays."""

    def __init__(self, class_colors: Dict[int, List[float]]):
        self.class_colors = class_colors

    def colorize(self, input_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
        input_normalized = self.normalizeInput(input_slice)
        colored = np.stack([input_normalized] * 3, axis=-1)

        for class_id, color in self.class_colors.items():
            mask = mask_slice == class_id
            if mask.any():
                colored = self.applyColorOverlay(colored, mask, color)

        return np.clip(colored, 0, 1)

    def normalizeInput(self, input_slice: np.ndarray) -> np.ndarray:
        return (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)

    def applyColorOverlay(self, image: np.ndarray, mask: np.ndarray,
                          color: List[float]) -> np.ndarray:
        for channel in range(3):
            image_channel = image[..., channel]
            image_channel[mask] = color[channel] * 0.3 + image_channel[mask] * 0.7
            image[..., channel] = image_channel
        return image


class SegmentationVisualizer(VisualizationStrategy):
    """Handles visualization of segmentation results."""

    def __init__(self, class_colors: Dict[int, List[float]] = None):
        self.mask_finder = MaskCenterFinder()
        self.colorizer = MaskColorizer(class_colors or self.defaultColors())

    def visualize(self, volumes: torch.Tensor, true_masks: torch.Tensor,
                  pred_logits: torch.Tensor, epoch: int, save_path: str,
                  phase: str, config_id: str, model_type: str,
                  launch_number: int) -> None:

        os.makedirs(save_path, exist_ok=True)

        batch_size = min(4, volumes.shape[0])
        volumes_np, true_masks_np, pred_masks_np = self.prepareData(
            volumes, true_masks, pred_logits)

        self.createVisualization(volumes_np, true_masks_np, pred_masks_np,
                                 batch_size, epoch, phase, config_id,
                                 model_type, launch_number, save_path)

    def prepareData(self, volumes: torch.Tensor, true_masks: torch.Tensor,
                    pred_logits: torch.Tensor) -> Tuple[np.ndarray, ...]:

        volumes_np = volumes.cpu().numpy()
        true_masks_np = true_masks.cpu().numpy()

        if true_masks_np.ndim == 5:
            true_masks_np = true_masks_np[:, 0]

        pred_masks_np = torch.softmax(pred_logits, dim=1).argmax(dim=1).cpu().numpy()

        return volumes_np, true_masks_np, pred_masks_np

    def createVisualization(self, volumes_np: np.ndarray, true_masks_np: np.ndarray,
                            pred_masks_np: np.ndarray, batch_size: int, epoch: int,
                            phase: str, config_id: str, model_type: str,
                            launch_number: int, save_path: str) -> None:

        fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
        if batch_size == 1:
            axes = axes.reshape(2, 1)

        for i in range(batch_size):
            slice_idx = self.mask_finder.find(true_masks_np[i])

            input_slice = volumes_np[i, 0, slice_idx]
            true_mask_slice = true_masks_np[i, slice_idx]
            pred_mask_slice = pred_masks_np[i, slice_idx]

            true_colored = self.colorizer.colorize(input_slice, true_mask_slice)
            pred_colored = self.colorizer.colorize(input_slice, pred_mask_slice)

            axes[0, i].imshow(true_colored)
            axes[0, i].set_title(f'GT Sample {i+1}')
            axes[0, i].axis('off')

            axes[1, i].imshow(pred_colored)
            axes[1, i].set_title(f'Pred Sample {i+1}')
            axes[1, i].axis('off')

        self.addLegend(fig)
        plt.suptitle(f'Epoch {epoch} - {phase.upper()} - Segmentation Results')
        plt.tight_layout()

        self.saveFigure(fig, epoch, phase, config_id, model_type,
                        launch_number, save_path)

    def addLegend(self, fig) -> None:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', label='Background'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.2, 0.8, 0.2], label='Healthy'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.9, 0.2], label='Partially injured'),
            plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.3, 0.3], label='Completely ruptured')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   bbox_to_anchor=(0.5, -0.05))

    def saveFigure(self, fig, epoch: int, phase: str, config_id: str,
                   model_type: str, launch_number: int, save_path: str) -> None:

        filename = f"{model_type}_{phase}_segmentation_epoch_{epoch}_hyperparams_{config_id}_{launch_number}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()

    def defaultColors(self) -> Dict[int, List[float]]:
        return {
            0: [0.5, 0.5, 0.5],  # Background
            1: [0.2, 0.8, 0.2],  # Healthy
            2: [0.9, 0.9, 0.2],  # Partially injured
            3: [0.9, 0.3, 0.3],  # Completely ruptured
        }
