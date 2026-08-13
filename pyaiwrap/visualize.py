import torch
import os
from torchvision.utils import make_grid, save_image
from typing import Optional, Dict, List, Tuple
from .transforms import labToRgb, labToRgbForVisualization, luminanceToLabRange
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# How many channels of the model input each INPUT_CHANNEL type actually describes. Anything
# beyond this is conditioning the model was given (segmentation encodings, edge maps), not
# picture, and must not reach the image conversions.
INPUT_CHANNEL_COUNTS: Dict[str, int] = {
    "luminance": 1, "R": 1, "G": 1, "B": 1, "LAB_A": 1, "LAB_B": 1,
    "AB": 2,
    "RGB": 3, "LAB": 3, "ab_to_3ch": 3,
}


class VisualizationStrategy(ABC):
    """Abstract base class for visualization strategies."""

    @abstractmethod
    def visualize(self, *args, **kwargs) -> None:
        pass


class ImageConverter:
    """Handles image conversion between different color spaces."""

    def __init__(self):
        self.single_channel_converters = {
            "RGB": self._convertRgb,
            "luminance": self._convertLuminance,
            "AB": self._convertAb,
            "LAB": self._convertLab,
            "R": self._convertSingleChannel,
            "G": self._convertSingleChannel,
            "B": self._convertSingleChannel,
        }

    def convert(self, images: torch.Tensor, channel_type: str,
                paired_images: Optional[torch.Tensor] = None,
                input_range: str = "zero_one") -> torch.Tensor:
        """Convert images to RGB using dispatch pattern."""

        if channel_type in ("LAB_A", "LAB_B"):
            rgb = self._convertLabChannel(images, channel_type, paired_images, input_range)
            return self._ensureBatchDimension(rgb)

        if self._isLAbPair(images, paired_images):
            rgb = self._convertLAbPair(images, paired_images, input_range)
            return self._ensureBatchDimension(rgb)

        converter = self.single_channel_converters.get(channel_type, self._convertDefault)
        rgb = converter(images, channel_type, input_range)
        return self._ensureBatchDimension(rgb)

    def _ensureBatchDimension(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        return rgb

    def _isLAbPair(self, images: torch.Tensor, paired_images: Optional[torch.Tensor]) -> bool:
        """Check if we have an L+AB pair."""
        return paired_images is not None and (
            (images.shape[1] == 1 and paired_images.shape[1] == 2) or
            (images.shape[1] == 2 and paired_images.shape[1] == 1)
        )

    def _convertLAbPair(self, images: torch.Tensor, paired_images: torch.Tensor,
                        input_range: str) -> torch.Tensor:
        """Convert L+AB pair to RGB."""
        if images.shape[1] == 1:  # images is L, paired_images is AB
            l_channel, ab_channels = images, paired_images
        else:  # images is AB, paired_images is L
            l_channel, ab_channels = paired_images, images

        if input_range == "zero_one":
            l_channel = l_channel * 100.0
            ab_channels = ab_channels * 254.0 - 127.0

        return labToRgb(l_channel, ab_channels)

    def _convertLabChannel(self, images: torch.Tensor, channel_type: str,
                           paired_images: Optional[torch.Tensor],
                           input_range: str) -> torch.Tensor:
        """Convert a single A or B channel to RGB, using paired L channel if available.

        paired_images (if given) must be an L channel already in Kornia's [0, 100] range.
        """
        if input_range == "zero_one":
            images = images * 254.0 - 127.0

        if paired_images is not None:
            l_channel = paired_images
        else:
            l_channel = torch.full_like(images, 50.0)

        zeros = torch.zeros_like(images)
        if channel_type == "LAB_A":
            ab_channels = torch.cat([images, zeros], dim=1)
        else:
            ab_channels = torch.cat([zeros, images], dim=1)

        return labToRgb(l_channel, ab_channels)

    def _convertRgb(self, images: torch.Tensor, channel_type: str, input_range: str) -> torch.Tensor:
        """RGB passes through unchanged."""
        return images

    def _convertLuminance(self, images: torch.Tensor, channel_type: str, input_range: str) -> torch.Tensor:
        """Convert luminance to grayscale RGB."""
        factor = 1.0 if input_range == "zero_one" else 1.0 / 100.0
        return (images * factor).repeat(1, 3, 1, 1)

    def _convertAb(self, images: torch.Tensor, channel_type: str, input_range: str) -> torch.Tensor:
        """Convert AB channels to false-color RGB."""
        if input_range == "zero_one":
            images = images * 254.0 - 127.0

        zeros = torch.zeros_like(images[:, 0:1])
        lab_3ch = torch.cat([zeros, images], dim=1)
        return labToRgbForVisualization(lab_3ch)

    def _convertLab(self, images: torch.Tensor, channel_type: str, input_range: str) -> torch.Tensor:
        """Convert LAB channels to RGB."""
        if input_range == "zero_one":
            lab = torch.zeros_like(images)
            lab[:, 0:1] = images[:, 0:1] * 100.0
            lab[:, 1:3] = images[:, 1:3] * 254.0 - 127.0
            return labToRgbForVisualization(lab)
        return labToRgbForVisualization(images)

    def _convertSingleChannel(self, images: torch.Tensor, channel_type: str, input_range: str) -> torch.Tensor:
        """Convert single channel (R, G, or B) to RGB."""
        zeros = torch.zeros_like(images)

        channel_map = {
            "R": (images, zeros, zeros),
            "G": (zeros, images, zeros),
            "B": (zeros, zeros, images)
        }

        return torch.cat(channel_map[channel_type], dim=1)

    def _convertDefault(self, images: torch.Tensor, channel_type: str, input_range: str) -> torch.Tensor:
        """Default conversion for unknown channel types."""
        return images.repeat(1, 3, 1, 1)


class RangeDetector:
    """Detects input range based on channel type and values."""

    @staticmethod
    def detect(images: torch.Tensor, channel_type: str) -> str:
        """Detect if images are in [0,1] range or Kornia's native range."""
        if channel_type in ["LAB", "AB", "LAB_A", "LAB_B", "luminance"]:
            if images.min() >= 0 and images.max() <= 1:
                return "zero_one"
            else:
                return "kornia"
        return "zero_one"


class ColorizationVisualizer(VisualizationStrategy):
    """Handles visualization of colorization/reconstruction tasks."""

    def __init__(self):
        self.converter = ImageConverter()
        self.range_detector = RangeDetector()
        self._setupConversionStrategies()

    def _setupConversionStrategies(self):
        """Setup strategy pattern for different conversion scenarios."""
        self.conversion_strategies = {
            ("luminance", "AB"): self._convertLuminanceAndAb
        }

        for channel in ["R", "G", "B"]:
            self.conversion_strategies[("luminance", channel)] = self._convertSingleChannel

        for channel in ["LAB_A", "LAB_B"]:
            self.conversion_strategies[("luminance", channel)] = self._convertLuminanceAndLabChannel

        self.default_strategy = self._convertGeneral

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

        images = self._prepareImages(
            original_images, modified_images, reconstructed_images, num_images, input_channel
        )

        ranges = self._getRanges(images, input_channel, target_channel, input_range)
        rgb_images = self._convertToRgb(images, input_channel, target_channel, ranges)
        self._saveVisualization(rgb_images, epoch, save_path, model_type,
                                launch_number, config_id, input_channel, target_channel)

    def _prepareImages(self, original: torch.Tensor, modified: torch.Tensor,
                       reconstructed: torch.Tensor, num_images: int,
                       input_channel: str = "RGB") -> Dict[str, torch.Tensor]:
        """Prepare and extract subset of images."""
        return {
            "original": original.detach().cpu()[:num_images],
            "modified": self._dropConditioningChannels(
                modified.detach().cpu()[:num_images], input_channel),
            "reconstructed": reconstructed.detach().cpu()[:num_images]
        }

    @staticmethod
    def _dropConditioningChannels(modified: torch.Tensor, input_channel: str) -> torch.Tensor:
        """Keep only the channels `input_channel` names, discarding conditioning stacked behind.

        A segmentation-conditioned model is fed [luminance, mask encoding]. Only the luminance
        is an image; the encoding is a description of one. Rendering it as though it were an
        image is what produced a 9-channel "grayscale" (3 channels tripled) and killed a run at
        the first visualisation epoch, five epochs of training after the mistake was made.

        An unknown channel type is passed through untouched: this must never be the reason a
        training run dies, and a visualisation that looks wrong is a far cheaper failure than
        one that stops the loop.
        """
        expected: Optional[int] = INPUT_CHANNEL_COUNTS.get(input_channel)
        if expected is None or modified.shape[1] <= expected:
            return modified
        return modified[:, :expected]

    def _getRanges(self, images: Dict[str, torch.Tensor], input_channel: str,
                   target_channel: str, input_range: str) -> Dict[str, str]:
        """Determine input ranges for each image set."""
        if input_range == "auto":
            return {
                "original": self.range_detector.detect(images["original"], target_channel),
                "modified": self.range_detector.detect(images["modified"], input_channel),
                "reconstructed": self.range_detector.detect(images["reconstructed"], target_channel)
            }
        return {k: input_range for k in ["original", "modified", "reconstructed"]}

    def _convertToRgb(self, images: Dict[str, torch.Tensor], input_channel: str,
                      target_channel: str, ranges: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Convert all image sets to RGB using appropriate strategy."""

        strategy_key = (input_channel, target_channel)
        strategy = self.conversion_strategies.get(strategy_key, self.default_strategy)
        if target_channel in ("R", "G", "B", "LAB_A", "LAB_B"):
            return strategy(images, ranges, input_channel, target_channel)
        else:
            return strategy(images, ranges)

    def _convertLuminanceAndAb(self, images: Dict[str, torch.Tensor],
                               ranges: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Strategy for L -> AB colorization."""
        modified = images["modified"]
        modified_range = ranges["modified"]

        # modified is the grey model input, so it needs the sRGB -> L* transfer, not a bare
        # x100; see transforms.grayToLightness. Distinct from _convertLab, whose "zero_one" L
        # really is L*/100 and is therefore still scaled arithmetically.
        l_channel = (luminanceToLabRange(modified, "srgb")
                     if modified_range == "zero_one" else modified)

        return {
            "original": self.converter.convert(
                images["original"], "AB", paired_images=l_channel,
                input_range=ranges["original"]
            ),
            "modified": self.converter.convert(modified, "luminance", input_range=modified_range),
            "reconstructed": self.converter.convert(
                images["reconstructed"], "AB", paired_images=l_channel,
                input_range=ranges["reconstructed"]
            )
        }

    def _convertLuminanceAndLabChannel(self, images: Dict[str, torch.Tensor],
                                       ranges: Dict[str, str],
                                       input_channel: str, target_channel: str) -> Dict[str, torch.Tensor]:
        """Strategy for L -> single A or B channel extraction."""
        modified = images["modified"]
        modified_range = ranges["modified"]

        # modified is the grey model input, so it needs the sRGB -> L* transfer, not a bare
        # x100; see transforms.grayToLightness. Distinct from _convertLab, whose "zero_one" L
        # really is L*/100 and is therefore still scaled arithmetically.
        l_channel = (luminanceToLabRange(modified, "srgb")
                     if modified_range == "zero_one" else modified)

        return {
            "original": self.converter.convert(
                images["original"], target_channel, paired_images=l_channel,
                input_range=ranges["original"]
            ),
            "modified": self.converter.convert(modified, "luminance", input_range=modified_range),
            "reconstructed": self.converter.convert(
                images["reconstructed"], target_channel, paired_images=l_channel,
                input_range=ranges["reconstructed"]
            )
        }

    def _convertSingleChannel(self, images: Dict[str, torch.Tensor],
                              ranges: Dict[str, str],
                              input_channel: str, target_channel: str) -> Dict[str, torch.Tensor]:
        return {
            "original": self.converter.convert(
                images["original"], target_channel, input_range=ranges["original"]
            ),
            "modified": self.converter.convert(
                images["modified"], input_channel, input_range=ranges["modified"]
            ),
            "reconstructed": self.converter.convert(
                images["reconstructed"], target_channel, input_range=ranges["reconstructed"]
            )
        }

    def _convertGeneral(self, images: Dict[str, torch.Tensor],
                        ranges: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Default strategy for general conversions."""
        return {
            "original": self.converter.convert(
                images["original"], "RGB", input_range=ranges["original"]
            ),
            "modified": self.converter.convert(
                images["modified"], "luminance", input_range=ranges["modified"]
            ),
            "reconstructed": self.converter.convert(
                images["reconstructed"], "RGB", input_range=ranges["reconstructed"]
            )
        }

    def _extractLuminance(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Extract luminance from RGB images."""
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_images.device).view(1, 3, 1, 1)
        return (rgb_images * weights).sum(dim=1, keepdim=True)

    def _saveVisualization(self, images: Dict[str, torch.Tensor], epoch: int,
                           save_path: str, model_type: str, launch_number: str,
                           config_id: str, input_channel: str, target_channel: str) -> None:
        """Save visualization grid to file."""

        os.makedirs(save_path, exist_ok=True)

        batch_sizes = [img.shape[0] for img in images.values()]
        min_batch = min(batch_sizes)

        grid_images = [img[:min_batch] for img in images.values()]
        comparison = torch.cat(grid_images, dim=0)
        grid = make_grid(comparison, nrow=min_batch, padding=2, normalize=False)

        filename = f'{model_type}_{launch_number}_{config_id}_epoch_{epoch:03d}_{input_channel}_to_{target_channel}.png'
        save_image(grid, os.path.join(save_path, filename))


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

    def __init__(self, class_colors: Dict[int, List[float]] = None):
        self.mask_finder = MaskCenterFinder()
        self.colorizer = MaskColorizer(class_colors or self.defaultColors())
        self.class_colors = class_colors

    def visualize(self, volumes: torch.Tensor, true_masks: torch.Tensor,
                  pred_logits: torch.Tensor, epoch: int, save_path: str,
                  phase: str, config_id: str, model_type: str,
                  launch_number: int) -> None:

        os.makedirs(save_path, exist_ok=True)

        batch_size = min(2, volumes.shape[0])
        volumes_np, true_masks_np, pred_masks_np = self.prepareData(
            volumes, true_masks, pred_logits)

        for sample_idx in range(batch_size):
            self.createFiveSliceVisualization(
                volumes_np[sample_idx, 0],
                true_masks_np[sample_idx],
                pred_masks_np[sample_idx],
                sample_idx, epoch, phase, config_id,
                model_type, launch_number, save_path
            )

    def prepareData(self, volumes: torch.Tensor, true_masks: torch.Tensor,
                    pred_logits: torch.Tensor) -> Tuple[np.ndarray, ...]:

        volumes_np = volumes.cpu().numpy()
        true_masks_np = true_masks.cpu().numpy()

        if true_masks_np.ndim == 5:
            true_masks_np = true_masks_np[:, 0]

        pred_masks_np = torch.softmax(pred_logits, dim=1).argmax(dim=1).cpu().numpy()

        return volumes_np, true_masks_np, pred_masks_np

    def createFiveSliceVisualization(self, volume: np.ndarray, true_mask: np.ndarray,
                                     pred_mask: np.ndarray, sample_idx: int, epoch: int,
                                     phase: str, config_id: str, model_type: str,
                                     launch_number: int, save_path: str) -> None:

        center_slice = self.mask_finder.find(true_mask)
        d = volume.shape[0]

        slice_indices = []
        for offset in [-2, -1, 0, 1, 2]:
            slice_idx = center_slice + offset
            slice_idx = max(0, min(d-1, slice_idx))
            slice_indices.append(slice_idx)

        slice_indices = list(dict.fromkeys(slice_indices))

        fig, axes = plt.subplots(2, len(slice_indices), figsize=(len(slice_indices) * 3, 6))
        if len(slice_indices) == 1:
            axes = axes.reshape(2, 1)

        for col, slice_idx in enumerate(slice_indices):
            input_slice = volume[slice_idx]
            true_mask_slice = true_mask[slice_idx]
            true_colored = self.colorizer.colorize(input_slice, true_mask_slice)

            axes[0, col].imshow(true_colored)
            axes[0, col].set_title(f'GT Slice {slice_idx}')
            axes[0, col].axis('off')

            if slice_idx == center_slice:
                for spine in axes[0, col].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)

        for col, slice_idx in enumerate(slice_indices):
            input_slice = volume[slice_idx]
            pred_mask_slice = pred_mask[slice_idx]
            pred_colored = self.colorizer.colorize(input_slice, pred_mask_slice)

            axes[1, col].imshow(pred_colored)
            axes[1, col].set_title(f'Pred Slice {slice_idx}')
            axes[1, col].axis('off')

            if slice_idx == center_slice:
                for spine in axes[1, col].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)

        self.addLegend(fig)

        plt.suptitle(f'Epoch {epoch} - {phase.upper()} - Sample {sample_idx+1}',
                     fontsize=14, y=1.02)

        plt.tight_layout()

        filename = f"{model_type}_{phase}_sample_{sample_idx}_segmentation_epoch_{epoch}_{config_id}_{launch_number}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

    def addLegend(self, fig) -> None:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.class_colors[0], label='Background'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.class_colors[1], label='Healthy'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.class_colors[2], label='Partially injured'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.class_colors[3], label='Completely ruptured')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   bbox_to_anchor=(0.5, -0.05), fontsize=10)

    def defaultColors(self) -> Dict[int, List[float]]:
        return {
            0: [0.5, 0.5, 0.5],
            1: [0.2, 0.8, 0.2],
            2: [0.9, 0.9, 0.2],
            3: [0.9, 0.3, 0.3],
        }
