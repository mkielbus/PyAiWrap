import math
import random

import torch
import torchvision.transforms.functional as TF
import kornia
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum


class ImageTransform(ABC):
    """Abstract base class for image transformations"""

    @abstractmethod
    def __call__(self, img):
        pass

    def _handleTensor(self, img):
        """Override in subclass to handle tensor input"""
        raise NotImplementedError

    def _handleNumpy(self, img):
        """Override in subclass to handle numpy input"""
        raise NotImplementedError

    def _handlePil(self, img):
        """Override in subclass to handle PIL input"""
        raise NotImplementedError


class PathAwareImageTransform(ABC):
    """A transform whose parameters depend on WHICH image it is given, not just its pixels.

    The cluster-version remap is the motivating case: the correspondence it applies is chosen
    from the source image's cluster and colour version, so it must be told the file the image
    came from. Kept separate from ImageTransform (rather than making every transform take a
    path) so plain photometric transforms stay usable anywhere; PairedImageFolder dispatches on
    the type, and ComposedTargetAugmentation lets the two kinds be chained.
    """

    @abstractmethod
    def __call__(self, img: Image.Image, image_path: str) -> Image.Image:
        pass


class ComposedTargetAugmentation(PathAwareImageTransform):
    """Chain target-side augmentations, passing the image path to those that need it."""

    def __init__(self, augmentations: List[Callable]) -> None:
        self.augmentations: List[Callable] = augmentations

    def __call__(self, img: Image.Image, image_path: str) -> Image.Image:
        for augmentation in self.augmentations:
            if isinstance(augmentation, PathAwareImageTransform):
                img = augmentation(img, image_path)
            else:
                img = augmentation(img)
        return img


class ToGrayscale(ImageTransform):
    """Convert image to grayscale"""

    def __init__(self, num_output_channels: int = 1):
        """
        Initialize grayscale transform.

        Args:
            num_output_channels: Number of output channels (1 or 3)
        """
        if num_output_channels not in [1, 3]:
            raise ValueError("num_output_channels must be 1 or 3")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """Apply grayscale conversion"""
        if isinstance(img, torch.Tensor):
            return self._handleTensor(img)
        elif isinstance(img, np.ndarray):
            return self._handleNumpy(img)
        elif isinstance(img, Image.Image):
            return self._handlePil(img)
        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def _handleTensor(self, img):
        """Convert tensor to grayscale"""
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1) if self.num_output_channels == 3 else img

        # Y = 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        gray = gray.unsqueeze(0)

        return gray.repeat(3, 1, 1) if self.num_output_channels == 3 else gray

    def _handleNumpy(self, img):
        """Convert numpy array to grayscale"""
        if img.ndim == 2:
            return np.stack([img] * 3, axis=2) if self.num_output_channels == 3 else img

        if img.ndim == 3:
            if img.shape[2] == 1:
                return np.repeat(img, 3, axis=2) if self.num_output_channels == 3 else img.squeeze(2)

            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            return np.stack([gray] * 3, axis=2) if self.num_output_channels == 3 else gray

        raise ValueError(f"Unsupported numpy array shape: {img.shape}")

    def _handlePil(self, img):
        """Convert PIL Image to grayscale"""
        return TF.to_grayscale(img, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ExtractChannel(ImageTransform):
    """Extract specific channel from RGB image"""

    def __init__(self, channel_index: int, num_output_channels: int = 1):
        """
        Initialize channel extraction.

        Args:
            channel_index: Index of channel to extract (0=Red, 1=Green, 2=Blue)
            num_output_channels: Number of output channels (1 or 3)
        """
        if channel_index not in [0, 1, 2]:
            raise ValueError("channel_index must be 0, 1, or 2")
        if num_output_channels not in [1, 3]:
            raise ValueError("num_output_channels must be 1 or 3")

        self.channel_index = channel_index
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """Extract specified channel"""
        if isinstance(img, torch.Tensor):
            return self._handleTensor(img)
        elif isinstance(img, np.ndarray):
            return self._handleNumpy(img)
        elif isinstance(img, Image.Image):
            return self._handlePil(img)
        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def _handleTensor(self, img):
        """Extract channel from tensor"""
        if img.shape[0] == 1:
            return img.repeat(3, 1, 1) if self.num_output_channels == 3 else img

        channel = img[self.channel_index:self.channel_index+1, :, :]
        return channel.repeat(3, 1, 1) if self.num_output_channels == 3 else channel

    def _handleNumpy(self, img):
        """Extract channel from numpy array"""
        if img.ndim == 2:
            return np.stack([img] * 3, axis=2) if self.num_output_channels == 3 else img

        if img.ndim == 3:
            if img.shape[2] == 1:
                return np.repeat(img, 3, axis=2) if self.num_output_channels == 3 else img.squeeze(2)

            channel = img[:, :, self.channel_index]
            return np.stack([channel] * 3, axis=2) if self.num_output_channels == 3 else channel

        raise ValueError(f"Unsupported numpy array shape: {img.shape}")

    def _handlePil(self, img):
        """Extract channel from PIL Image"""
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img)
        channel = img_array[:, :, self.channel_index]

        if self.num_output_channels == 1:
            return Image.fromarray(channel, mode='L')
        else:
            channel_3ch = np.stack([channel] * 3, axis=2)
            return Image.fromarray(channel_3ch.astype(np.uint8), mode='RGB')

    def __repr__(self):
        channel_names = ['Red', 'Green', 'Blue']
        return (f"{self.__class__.__name__}(channel={channel_names[self.channel_index]}, "
                f"num_output_channels={self.num_output_channels})")


class ExtractChannelTo3Channel(ImageTransform):
    """Extract channel and create 3-channel image with zeros in other channels"""

    def __init__(self, channel_index: int):
        """
        Initialize channel extraction to 3-channel format.

        Args:
            channel_index: Index of channel to extract (0=Red, 1=Green, 2=Blue)
        """
        if channel_index not in [0, 1, 2]:
            raise ValueError("channel_index must be 0, 1, or 2")
        self.channel_index = channel_index

    def __call__(self, img):
        """Extract channel and create 3-channel output"""
        if isinstance(img, torch.Tensor):
            return self._handleTensor(img)
        elif isinstance(img, Image.Image):
            return self._handlePil(img)
        else:
            raise TypeError(f"Unsupported type: {type(img)}. Use PIL Image or Tensor.")

    def _handleTensor(self, img):
        """Extract channel from tensor and create 3-channel output"""
        if img.shape[0] == 1:
            result = torch.zeros(3, img.shape[1], img.shape[2])
            result[self.channel_index:self.channel_index+1, :, :] = img
            return result

        channel = img[self.channel_index:self.channel_index+1, :, :]
        result = torch.zeros_like(img)
        result[self.channel_index:self.channel_index+1, :, :] = channel
        return result

    def _handlePil(self, img):
        """Extract channel from PIL Image and create 3-channel tensor"""
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img)
        channel = img_array[:, :, self.channel_index]

        result = np.zeros_like(img_array)
        result[:, :, self.channel_index] = channel

        return torch.from_numpy(result).permute(2, 0, 1).float() / 255.0

    def _handleNumpy(self, img):
        """Not implemented for this transform"""
        raise NotImplementedError("ExtractChannelTo3Channel does not support numpy arrays")

    def __repr__(self):
        channel_names = ['Red', 'Green', 'Blue']
        return f"{self.__class__.__name__}(channel={channel_names[self.channel_index]})"


class ExtractRedChannel(ExtractChannel):
    """Extract red channel from RGB image"""
    def __init__(self, num_output_channels: int = 1):
        super().__init__(channel_index=0, num_output_channels=num_output_channels)


class ExtractGreenChannel(ExtractChannel):
    """Extract green channel from RGB image"""
    def __init__(self, num_output_channels: int = 1):
        super().__init__(channel_index=1, num_output_channels=num_output_channels)


class ExtractBlueChannel(ExtractChannel):
    """Extract blue channel from RGB image"""
    def __init__(self, num_output_channels: int = 1):
        super().__init__(channel_index=2, num_output_channels=num_output_channels)


class ExtractRedChannelTo3Channel(ExtractChannelTo3Channel):
    """Extract red channel and create 3-channel image with format [red, 0, 0]"""
    def __init__(self):
        super().__init__(channel_index=0)


class ExtractGreenChannelTo3Channel(ExtractChannelTo3Channel):
    """Extract green channel and create 3-channel image with format [0, green, 0]"""
    def __init__(self):
        super().__init__(channel_index=1)


class ExtractBlueChannelTo3Channel(ExtractChannelTo3Channel):
    """Extract blue channel and create 3-channel image with format [0, 0, blue]"""
    def __init__(self):
        super().__init__(channel_index=2)


class RGBToLAB(ImageTransform):
    """Convert RGB image to LAB color space using Kornia conversion"""

    def __init__(self):
        """Initialize RGB to LAB conversion"""
        super().__init__()

    def __call__(self, img):
        """Convert RGB image to LAB color space and return as tensor"""
        if isinstance(img, torch.Tensor):
            return self._handleTensor(img)
        elif isinstance(img, np.ndarray):
            return self._handleNumpy(img)
        elif isinstance(img, Image.Image):
            return self._handlePil(img)
        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def _handleTensor(self, img):
        """Convert tensor from RGB to LAB using Kornia"""
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Add batch dimension

        if img.shape[-3] != 3:
            raise ValueError("Input tensor must have 3 channels for RGB to LAB conversion")

        # Ensure RGB is in [0, 1] range for Kornia conversion
        if img.max() > 1.0:
            img = img / 255.0

        # Kornia returns: L: [0, 100], A: [-127, 127], B: [-127, 127] (approximately)
        lab = kornia.color.rgb_to_lab(img)

        # Remove batch dimension if input was 3D
        if lab.dim() == 4 and lab.shape[0] == 1:
            lab = lab.squeeze(0)

        return lab

    def _handleNumpy(self, img):
        """Convert numpy array from RGB to LAB and return as tensor"""
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Input array must have 3 channels for RGB to LAB conversion")

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return self._handleTensor(img_tensor)

    def _handlePil(self, img):
        """Convert PIL Image from RGB to LAB and return as tensor"""
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_tensor = transforms.ToTensor()(img)
        return self._handleTensor(img_tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExtractABChannels(ImageTransform):
    """Extract A and B channels from LAB color space - expects LAB input in Kornia's ranges"""

    def __init__(self, num_output_channels: int = 2):
        """
        Initialize AB channels extraction.

        IMPORTANT: Expects input to already be in LAB color space with Kornia's ranges:
        L: [0, 100], A: [-127, 127], B: [-127, 127]

        Args:
            num_output_channels: Number of output channels (2 or 3)
        """
        super().__init__()
        if num_output_channels not in [2, 3]:
            raise ValueError("num_output_channels must be 2 or 3")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Extract A and B channels from LAB tensor.

        Expects input to be LAB tensor from RGBToLAB transform.
        """
        # Direct tensor extraction - assumes input is already LAB
        if isinstance(img, torch.Tensor):
            return self._extract_ab_channels(img)
        else:
            # If input is not tensor, it's probably not in LAB space
            raise TypeError("ExtractABChannels expects LAB tensor input. Use RGBToLAB transform first.")

    def _extract_ab_channels(self, lab_tensor):
        """Extract AB channels from LAB tensor"""
        if lab_tensor.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {lab_tensor.dim()}")

        if lab_tensor.shape[0] != 3:
            raise ValueError(f"Input tensor must have 3 channels (LAB), got {lab_tensor.shape[0]}")

        # Simple tensor slicing to get AB channels
        # L: [0, 100], A: [-127, 127], B: [-127, 127]
        ab_channels = lab_tensor[1:3, :, :]

        if self.num_output_channels == 3:
            # Create 3-channel output with zeros in L channel
            zeros = torch.zeros_like(ab_channels[0:1, :, :])
            return torch.cat([zeros, ab_channels], dim=0)
        else:
            return ab_channels

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ExtractLABChannel(ImageTransform):
    """Extract a single A or B channel from LAB color space - expects LAB input in Kornia's ranges"""

    def __init__(self, channel_index: int, num_output_channels: int = 1):
        """
        Initialize single LAB channel extraction.

        IMPORTANT: Expects input to already be in LAB color space with Kornia's ranges:
        L: [0, 100], A: [-127, 127], B: [-127, 127]

        The extracted channel keeps Kornia's native [-127, 127] range, so the
        regressing network must have an unbounded output (no sigmoid).

        Args:
            channel_index: Index of channel to extract (1=A, 2=B)
            num_output_channels: Number of output channels (1 or 3)
        """
        super().__init__()
        if channel_index not in [1, 2]:
            raise ValueError("channel_index must be 1 (A) or 2 (B)")
        if num_output_channels not in [1, 3]:
            raise ValueError("num_output_channels must be 1 or 3")
        self.channel_index = channel_index
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """Extract a single A or B channel from LAB tensor"""
        if isinstance(img, torch.Tensor):
            return self._handleTensor(img)
        else:
            raise TypeError("ExtractLABChannel expects LAB tensor input. Use RGBToLAB transform first.")

    def _handleTensor(self, lab_tensor):
        """Extract the channel from LAB tensor in Kornia's native range"""
        if lab_tensor.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {lab_tensor.dim()}")

        if lab_tensor.shape[0] != 3:
            raise ValueError(f"Input tensor must have 3 channels (LAB), got {lab_tensor.shape[0]}")

        channel = lab_tensor[self.channel_index:self.channel_index + 1, :, :]

        return channel.repeat(3, 1, 1) if self.num_output_channels == 3 else channel

    def __repr__(self):
        channel_names = {1: 'A', 2: 'B'}
        return (f"{self.__class__.__name__}(channel={channel_names[self.channel_index]}, "
                f"num_output_channels={self.num_output_channels})")


class ExtractLABAChannel(ExtractLABChannel):
    """Extract A channel from LAB color space in Kornia's native range"""
    def __init__(self, num_output_channels: int = 1):
        super().__init__(channel_index=1, num_output_channels=num_output_channels)


class ExtractLABBChannel(ExtractLABChannel):
    """Extract B channel from LAB color space in Kornia's native range"""
    def __init__(self, num_output_channels: int = 1):
        super().__init__(channel_index=2, num_output_channels=num_output_channels)


class ExtractABChannelsTo3Channel(ImageTransform):
    """Extract A and B channels and create 3-channel LAB tensor with zeros in L channel"""

    def __init__(self):
        """Initialize AB channels extraction to 3-channel LAB format"""
        super().__init__()

    def __call__(self, img):
        """Extract AB channels and create 3-channel LAB tensor"""
        # Direct tensor extraction - assumes input is already LAB
        if isinstance(img, torch.Tensor):
            return self._extract_to_3channel(img)
        else:
            raise TypeError("ExtractABChannelsTo3Channel expects LAB tensor input. Use RGBToLAB transform first.")

    def _extract_to_3channel(self, lab_tensor):
        """Extract AB channels from tensor and create 3-channel LAB output"""
        if lab_tensor.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {lab_tensor.dim()}")

        if lab_tensor.shape[0] != 3:
            raise ValueError(f"Input tensor must have 3 channels (LAB), got {lab_tensor.shape[0]}")

        # Extract AB channels
        ab_channels = lab_tensor[1:3, :, :]

        # Create 3-channel output with zeros in L channel
        zeros = torch.zeros_like(ab_channels[0:1, :, :])
        return torch.cat([zeros, ab_channels], dim=0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def labToRgb(lChannel, abChannels):
    """
    Convert L and AB channels to RGB using Kornia's built-in conversion.
    Expects channels in Kornia's native ranges.

    Args:
        lChannel: L channel in [0, 100] range (Kornia's native range)
        abChannels: AB channels in [-127, 127] range (Kornia's native range)

    Returns:
        RGB tensor in [0,1] range
    """
    # Only strip the batch dimension if it was added here, so batched
    # size-1 inputs keep their rank (loss/LPIPS expect 4D in -> 4D out)
    added_batch_dim = lChannel.dim() == 3
    if lChannel.dim() == 3:
        lChannel = lChannel.unsqueeze(0)
    if abChannels.dim() == 3:
        abChannels = abChannels.unsqueeze(0)

    # Combine L and AB channels - both in Kornia's native ranges
    lab = torch.cat([lChannel, abChannels], dim=1)

    rgb = kornia.color.lab_to_rgb(lab)

    if added_batch_dim:
        rgb = rgb.squeeze(0)

    return rgb


def labToRgbForVisualization(labTensor):
    """
    Convert LAB tensor to RGB tensor for visualization purposes.
    Uses Kornia's built-in conversion.

    Args:
        labTensor: LAB tensor of shape (batch_size, 3, H, W) in Kornia format
                  L: [0,100], A: [-127,127], B: [-127,127]

    Returns:
        RGB tensor in range [0, 1]
    """
    added_batch_dim = labTensor.dim() == 3
    if added_batch_dim:
        labTensor = labTensor.unsqueeze(0)

    rgb = kornia.color.lab_to_rgb(labTensor)

    if added_batch_dim:
        rgb = rgb.squeeze(0)

    return rgb


class ChannelType(Enum):
    """Enum representing different channel types."""
    RGB = "RGB"
    LAB = "LAB"
    AB = "AB"
    AB_TO_3CH = "ab_to_3ch"
    LAB_A = "LAB_A"
    LAB_B = "LAB_B"
    LUMINANCE = "luminance"
    R = "R"
    G = "G"
    B = "B"


class TransformCreator(ABC):
    """Abstract factory interface for creating transform compositions."""

    @abstractmethod
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        pass


class RGBTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])


class LABTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB()
        ])


class ABTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        if output_channels not in [2, 3]:
            raise ValueError("output_channels must be 2 or 3 for 'ab' channel_type")
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB(),
            ExtractABChannels(num_output_channels=output_channels)
        ])


class ABTo3ChannelTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB(),
            ExtractABChannelsTo3Channel()
        ])


class LABAChannelTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB(),
            ExtractLABAChannel(num_output_channels=output_channels)
        ])


class LABBChannelTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB(),
            ExtractLABBChannel(num_output_channels=output_channels)
        ])


class LuminanceTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        if not is_input:
            raise ValueError("luminance can only be used for input channels, not target channels")
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ToGrayscale(num_output_channels=output_channels),
            transforms.ToTensor()
        ])


class RedChannelTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractRedChannel(num_output_channels=output_channels)
        ])


class GreenChannelTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractGreenChannel(num_output_channels=output_channels)
        ])


class BlueChannelTransformCreator(TransformCreator):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractBlueChannel(num_output_channels=output_channels)
        ])


class ChannelTransformFactory:
    """Main creator class that uses the abstract factory pattern."""

    _creators: Dict[ChannelType, TransformCreator] = {
        ChannelType.RGB: RGBTransformCreator(),
        ChannelType.LAB: LABTransformCreator(),
        ChannelType.AB: ABTransformCreator(),
        ChannelType.AB_TO_3CH: ABTo3ChannelTransformCreator(),
        ChannelType.LAB_A: LABAChannelTransformCreator(),
        ChannelType.LAB_B: LABBChannelTransformCreator(),
        ChannelType.LUMINANCE: LuminanceTransformCreator(),
        ChannelType.R: RedChannelTransformCreator(),
        ChannelType.G: GreenChannelTransformCreator(),
        ChannelType.B: BlueChannelTransformCreator()
    }

    @classmethod
    def getTransform(cls, channel_type: str, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        """Get the appropriate transform for the given channel type."""
        try:
            channel_enum = ChannelType(channel_type)
            creator = cls._creators[channel_enum]
            return creator.createTransform(image_size, output_channels, is_input)
        except (KeyError, ValueError):
            raise ValueError(f"channel_type must be one of {[c.value for c in ChannelType]}, got '{channel_type}'")


class AspectPreservingRandomResizedCrop(ImageTransform):
    """Random-area, random-position crop that keeps the source image's aspect ratio.

    The crop is a scaled-down copy of the frame's own shape: with the crop's aspect ratio
    set to the image's, a crop of area fraction `f` is exactly `W*sqrt(f)` by `H*sqrt(f)`,
    so it always fits and needs no fallback.

    Why the aspect ratio must not be randomised: the crop is then squared off to
    `image_size`, and validation/inference square off the WHOLE image the same way. The
    anisotropic stretching a given image receives at validation is therefore exactly `H/W`,
    a quantity that is known per image -- a random ratio band can only approximate it, and
    measured on this dataset it matches for ~3% of samples. Preserving the aspect makes
    training's distortion identical to validation's for every image, while the random area
    and position keep all of the regularisation. Validation is then simply this transform
    at `f = 1` without the flip.
    """

    def __init__(self, image_size: int, scale_min: float = 0.6, scale_max: float = 1.0) -> None:
        if not 0.0 < scale_min <= scale_max <= 1.0:
            raise ValueError(
                f"require 0 < scale_min <= scale_max <= 1, got ({scale_min}, {scale_max})"
            )
        self.image_size: int = image_size
        self.scale_min: float = scale_min
        self.scale_max: float = scale_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"AspectPreservingRandomResizedCrop expects a PIL image, got {type(img)}")

        width, height = img.size
        side_fraction: float = math.sqrt(random.uniform(self.scale_min, self.scale_max))
        crop_width: int = max(1, min(width, round(width * side_fraction)))
        crop_height: int = max(1, min(height, round(height * side_fraction)))

        left: int = random.randint(0, width - crop_width)
        top: int = random.randint(0, height - crop_height)

        return TF.resized_crop(img, top, left, crop_height, crop_width,
                               [self.image_size, self.image_size])


def createSharedGeometricAugmentation(image_size: int,
                                      flip_probability: float = 0.5,
                                      crop_scale_min: float = 0.6,
                                      crop_scale_max: float = 1.0,
                                      ratio_min: Optional[float] = None,
                                      ratio_max: Optional[float] = None) -> transforms.Compose:
    """Build the train-time paired geometric augmentation.

    The returned transform is applied once per image (on the PIL image) and shared by
    the input and target transforms via PairedImageFolder(shared_augmentation=...), so
    input and target stay pixel aligned. It replaces the deterministic square resize with
    a random-resized crop plus horizontal flip; no rotation (border artifacts).

    By default the crop keeps the source image's aspect ratio
    (AspectPreservingRandomResizedCrop), which makes the anisotropic distortion of squaring
    the crop off identical to the one validation applies to that same image. Passing an
    explicit ratio band restores the older torchvision RandomResizedCrop behaviour, which
    randomises the crop's aspect and therefore distorts training differently from
    validation; it exists to reproduce runs configured before the default changed
    (v3/v4 used 0.85-1.18) and is not recommended for new configs.
    """
    if (ratio_min is None) != (ratio_max is None):
        raise ValueError("ratio_min and ratio_max must be set together, or both left as None")

    if ratio_min is None:
        crop: ImageTransform = AspectPreservingRandomResizedCrop(
            image_size, scale_min=crop_scale_min, scale_max=crop_scale_max
        )
    else:
        crop = transforms.RandomResizedCrop(
            image_size,
            scale=(crop_scale_min, crop_scale_max),
            ratio=(ratio_min, ratio_max)
        )

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=flip_probability),
        crop
    ])


# Empirical mean_chroma band (p2..p98) of the dataset, measured in OpenCV LAB units by
# analysis/phase0_budget.py over analysis_results/image_colors.csv (Phase 0.3). Used to
# keep chroma scaling inside the colors the dataset actually contains.
CHROMA_BAND_LOW: float = 9.30
CHROMA_BAND_HIGH: float = 43.57


class ChromaJitter(ImageTransform):
    """BigColor-style chroma scaling of a target image, bounded to the dataset band.

    Scales LAB chroma (ab <- s * ab, hue unchanged) by a per-call scalar s, applied with
    probability `probability`. s is drawn from [chroma_min, chroma_max] but clamped per image
    so the resulting mean chroma stays inside [chroma_band_low, chroma_band_high] (the p2..p98
    band of mean_chroma; Phase 0.3), so an already-saturated image is never pushed past what
    the dataset contains. Chroma is measured in OpenCV LAB units, matching
    analysis/extract_colors.py. Target-side only, which keeps the luminance input pristine.
    """

    def __init__(self, probability: float = 0.5,
                 chroma_min: float = 1.0, chroma_max: float = 1.5,
                 chroma_band_low: float = CHROMA_BAND_LOW,
                 chroma_band_high: float = CHROMA_BAND_HIGH,
                 reference_max_side: int = 256) -> None:
        self.probability: float = probability
        self.chroma_min: float = chroma_min
        self.chroma_max: float = chroma_max
        self.chroma_band_low: float = chroma_band_low
        self.chroma_band_high: float = chroma_band_high
        # The band was measured at max_side 256 (extract_colors.py); measure chroma at the
        # same reference resolution so the bound holds regardless of the input image size.
        self.reference_max_side: int = reference_max_side

    def __call__(self, img: Image.Image) -> Image.Image:
        if isinstance(img, Image.Image):
            return self._handlePil(img)
        raise TypeError(f"ChromaJitter expects a PIL image, got {type(img)}")

    def _measureMeanChroma(self, rgb: np.ndarray) -> float:
        """OpenCV-LAB mean chroma at the band's reference resolution."""
        height, width = rgb.shape[:2]
        longest_side: int = max(height, width)
        if longest_side > self.reference_max_side:
            scale: float = self.reference_max_side / longest_side
            rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        lab: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        a: np.ndarray = lab[..., 1] - 128.0
        b: np.ndarray = lab[..., 2] - 128.0
        return float(np.sqrt(a * a + b * b).mean())

    def _sampleScale(self, current_chroma: float) -> float:
        """Scale factor honouring [chroma_min, chroma_max] and the empirical band."""
        scale_low: float = max(self.chroma_min, self.chroma_band_low / current_chroma)
        scale_high: float = min(self.chroma_max, self.chroma_band_high / current_chroma)
        if scale_low <= scale_high:
            return random.uniform(scale_low, scale_high)
        if current_chroma < self.chroma_band_low:      # too desaturated: push up as far as allowed
            return self.chroma_max
        return 1.0                                     # already above band: leave unchanged

    def _handlePil(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return img
        rgb: np.ndarray = np.asarray(img.convert("RGB"))
        current_chroma: float = self._measureMeanChroma(rgb)
        if current_chroma < 1e-6:                      # achromatic: nothing to scale
            return img
        scale: float = self._sampleScale(current_chroma)
        lab: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 1] = np.clip(128.0 + scale * (lab[..., 1] - 128.0), 0.0, 255.0)
        lab[..., 2] = np.clip(128.0 + scale * (lab[..., 2] - 128.0), 0.0, 255.0)
        scaled_rgb: np.ndarray = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return Image.fromarray(scaled_rgb, mode="RGB")


def createChromaJitter(probability: float = 0.5,
                       chroma_min: float = 1.0,
                       chroma_max: float = 1.5) -> ChromaJitter:
    """Build the target-side chroma-scaling augmentation (band bounds from Phase 0.3)."""
    return ChromaJitter(probability=probability, chroma_min=chroma_min, chroma_max=chroma_max)


class LumaJitter(ImageTransform):
    """Mild tone perturbation of the model INPUT only: gamma, then contrast, then brightness.

    A regulariser against instance memorisation, not a domain-matching measure -- validation
    and test images are clean, so this deliberately makes training harder than inference. It
    exists because rgb_merge_unet_v5 reaches 0.0975 LPIPS on the clean training set against
    0.1357 on validation, a 39% gap that says the network is keying on properties of
    individual training images; perturbing the exact luminance statistics breaks that index.

    Deliberately restricted to tone-curve changes, with no noise and no compression artefacts:
    the corpus genuinely contains several digitisations of the same artwork under different
    tone curves (analysis_results/phase0_v3/version_inventory.csv), so this perturbation stays
    inside the distribution, whereas sensor noise or JPEG blocking would not and would push
    the model to discard the fine luminance detail it needs.

    The target is untouched, which keeps the colours being learnt exactly the true ones.
    """

    def __init__(self, probability: float = 0.5,
                 gamma_min: float = 0.85, gamma_max: float = 1.18,
                 contrast: float = 0.08, brightness: float = 0.08) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if not 0.0 < gamma_min <= gamma_max:
            raise ValueError(f"require 0 < gamma_min <= gamma_max, got ({gamma_min}, {gamma_max})")
        if not 0.0 <= contrast < 1.0:
            raise ValueError(f"contrast must be in [0, 1), got {contrast}")
        if not 0.0 <= brightness < 1.0:
            raise ValueError(f"brightness must be in [0, 1), got {brightness}")

        self.probability: float = probability
        self.gamma_min: float = gamma_min
        self.gamma_max: float = gamma_max
        self.contrast: float = contrast
        self.brightness: float = brightness

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"LumaJitter expects a PIL image, got {type(img)}")
        if random.random() >= self.probability:
            return img

        gamma: float = random.uniform(self.gamma_min, self.gamma_max)
        contrast_factor: float = 1.0 + random.uniform(-self.contrast, self.contrast)
        brightness_factor: float = 1.0 + random.uniform(-self.brightness, self.brightness)

        # adjust_gamma/adjust_* keep the image in PIL form, so this composes with the rest of
        # the PIL-stage pipeline without a tensor round trip.
        jittered: Image.Image = TF.adjust_gamma(img, gamma)
        jittered = TF.adjust_contrast(jittered, contrast_factor)
        return TF.adjust_brightness(jittered, brightness_factor)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(probability={self.probability}, "
                f"gamma=({self.gamma_min}, {self.gamma_max}), contrast={self.contrast}, "
                f"brightness={self.brightness})")


def createLumaJitter(probability: float = 0.5,
                     gamma_min: float = 0.85, gamma_max: float = 1.18,
                     contrast: float = 0.08, brightness: float = 0.08) -> LumaJitter:
    """Build the input-side tone-jitter augmentation."""
    return LumaJitter(probability=probability, gamma_min=gamma_min, gamma_max=gamma_max,
                      contrast=contrast, brightness=brightness)


# Color-classification bands, mirroring analysis/extract_colors.py (and
# analysis_results/extract_colors_config.json) so the version remap operates on exactly the
# same named colors the version labels were derived from. Hue is in degrees (0-360).
REMAP_SATURATION_THRESHOLD: float = 0.20
REMAP_BROWN_HUE: Tuple[float, float] = (15.0, 50.0)
REMAP_BROWN_V: float = 0.55
# Shadows are near-black regions that keep their colour under any real-world recolour (repaint a
# green lawn brown and its shade stays near-black, it does not become brown). Pixels darker than
# this are excluded from the remap, ramping in over REMAP_FEATHER_VALUE so there is no seam. The
# floor is extract_colors.py's black threshold, so "protected shadow" means exactly "a pixel the
# colour taxonomy calls black". Raising it further starts suppressing legitimate repaints of dark
# OBJECTS (a dark red barn -> yellow), which leaves them two-tone; verified in
# analysis/verify_shadow_protection.py.
REMAP_SHADOW_VALUE: float = 0.20
REMAP_FEATHER_VALUE: float = 0.10
# Which hue band a pixel falls in is a NOISY per-pixel decision on a flat surface: measured on
# real images, 62-82% of the in-band pixels of a speckling region sit within 10 deg of a band
# edge, so a narrow feather makes neighbouring pixels of one surface land on opposite sides and
# the remap comes out salt-and-pepper. Widen the edge feather and spatially smooth the blend
# weight (median = drop isolated pixels, box = soften the rest); the hard protections (achromatic
# threshold, shadow floor) are re-applied AFTER smoothing so they can never be blurred away.
REMAP_FEATHER_DEG: float = 15.0
REMAP_WEIGHT_SMOOTHING: int = 3
REMAP_HUE_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("red", 345.0, 15.0), ("orange", 15.0, 45.0), ("yellow", 45.0, 70.0),
    ("green", 70.0, 165.0), ("cyan", 165.0, 200.0), ("blue", 200.0, 255.0),
    ("purple", 255.0, 290.0), ("magenta", 290.0, 320.0), ("pink", 320.0, 345.0),
)
ACHROMATIC_COLORS: frozenset = frozenset({"black", "gray", "white"})
CHROMATIC_HUE_BAND: Dict[str, Tuple[float, float]] = (
    {name: (start, end) for name, start, end in REMAP_HUE_BINS} | {"brown": REMAP_BROWN_HUE}
)


@dataclass
class RemapTarget:
    """Where a source color maps to, plus the target color's S/V statistics (as observed in
    the cluster) used for S/V matching so e.g. gold->bronze darkens correctly."""
    target_color: str
    saturation_mean: float
    saturation_std: float
    value_mean: float
    value_std: float


class ClusterVersionRemap(ImageTransform):
    """Cluster-conditioned color-version remap (Phase 1b L5b).

    Remaps a whole color version to another version observed in the same cluster, one global
    LUT per image: chromatic source colors are hue-mapped band->band (relative position within
    the band preserved), chromatic->achromatic is a pure desaturation, and achromatic source
    pixels (neutral backgrounds) are never touched. Dark pixels (value < shadow_value) are never
    touched either: shadows stay near-black under a recolour instead of being tinted and lifted
    toward the target's mean value. Mapped pixels' saturation/value are matched to the target
    color's S/V statistics. Band edges, the achromatic saturation threshold and the shadow floor
    are feathered to avoid seams and chroma-noise flicker. Colors are classified with the exact
    bands from analysis/extract_colors.py. Target-side only.

    correspondence maps each source color name to a RemapTarget; entries whose target equals the
    source, and any achromatic source, are treated as identity.
    """

    def __init__(self, correspondence: Dict[str, RemapTarget], probability: float = 0.5,
                 saturation_threshold: float = REMAP_SATURATION_THRESHOLD,
                 feather_deg: float = REMAP_FEATHER_DEG, feather_saturation: float = 0.05,
                 shadow_value: float = REMAP_SHADOW_VALUE,
                 feather_value: float = REMAP_FEATHER_VALUE,
                 weight_smoothing: int = REMAP_WEIGHT_SMOOTHING,
                 sv_ratio_bounds: Tuple[float, float] = (0.5, 1.5)) -> None:
        if weight_smoothing not in (0, 3, 5):
            raise ValueError("weight_smoothing must be 0 (off), 3 or 5 "
                             f"(cv2.medianBlur float32 limit), got {weight_smoothing}")
        self.correspondence: Dict[str, RemapTarget] = correspondence
        self.probability: float = probability
        self.saturation_threshold: float = saturation_threshold
        self.feather_deg: float = feather_deg
        self.feather_saturation: float = feather_saturation
        self.shadow_value: float = shadow_value
        self.feather_value: float = feather_value
        self.weight_smoothing: int = weight_smoothing
        # Bounds on the S/V std-matching ratio. Unbounded ratios blow up when a source region
        # is nearly uniform (tiny sample std), turning matching into salt-and-pepper noise.
        self.sv_ratio_bounds: Tuple[float, float] = sv_ratio_bounds

    def __call__(self, img: Image.Image) -> Image.Image:
        if isinstance(img, Image.Image):
            return self._handlePil(img)
        raise TypeError(f"ClusterVersionRemap expects a PIL image, got {type(img)}")

    @staticmethod
    def _inBand(hue: np.ndarray, start: float, end: float) -> np.ndarray:
        if start < end:
            return (hue >= start) & (hue < end)
        return (hue >= start) | (hue < end)      # red wraps through 360

    def _sourceMask(self, color: str, hue: np.ndarray, achromatic: np.ndarray,
                    brown_mask: np.ndarray) -> np.ndarray:
        if color == "brown":
            return brown_mask
        start, end = CHROMATIC_HUE_BAND[color]
        return (~achromatic) & self._inBand(hue, start, end) & (~brown_mask)

    def _protectionGate(self, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Hard per-pixel protections as a soft gate in [0,1]: neutral pixels (below the
        achromatic threshold) and shadows (below the value floor) are never remapped. Applied
        AFTER any spatial smoothing so smoothing cannot leak weight into a protected pixel."""
        weight_saturation: np.ndarray = np.clip(
            (saturation - self.saturation_threshold) / max(self.feather_saturation, 1e-6), 0.0, 1.0)
        weight_value: np.ndarray = np.clip(
            (value - self.shadow_value) / max(self.feather_value, 1e-6), 0.0, 1.0)
        return weight_saturation * weight_value

    def _bandWeight(self, color: str, hue: np.ndarray) -> np.ndarray:
        """Soft membership of the source hue band, fading towards the band edges so a surface
        whose hue jitters across an edge is not split pixel-by-pixel."""
        start, end = CHROMATIC_HUE_BAND[color]
        distance_to_edge: np.ndarray = np.minimum((hue - start) % 360.0, (end - hue) % 360.0)
        return np.clip(distance_to_edge / max(self.feather_deg, 1e-6), 0.0, 1.0)

    def _smoothWeight(self, weight: np.ndarray) -> np.ndarray:
        """Spatially de-speckle the blend weight: band membership is a noisy per-pixel decision,
        but a real recolour applies to whole surfaces. Median drops isolated pixels, the box blur
        softens what remains."""
        if self.weight_smoothing < 3:
            return weight
        smoothed: np.ndarray = cv2.medianBlur(weight.astype(np.float32), self.weight_smoothing)
        return cv2.blur(smoothed, (self.weight_smoothing, self.weight_smoothing)).astype(np.float64)

    @staticmethod
    def _remapHue(hue: np.ndarray, source_band: Tuple[float, float],
                  target_band: Tuple[float, float]) -> np.ndarray:
        source_width: float = (source_band[1] - source_band[0]) % 360.0 or 360.0
        target_width: float = (target_band[1] - target_band[0]) % 360.0 or 360.0
        position: np.ndarray = ((hue - source_band[0]) % 360.0) / source_width
        return (target_band[0] + position * target_width) % 360.0

    def _matchMoments(self, values: np.ndarray, sample: np.ndarray,
                      target_mean: float, target_std: float) -> np.ndarray:
        sample_mean: float = float(sample.mean())
        sample_std: float = float(sample.std())
        if sample_std < 1e-4:
            return values - sample_mean + target_mean          # uniform source: shift mean only
        low, high = self.sv_ratio_bounds
        ratio: float = min(max(target_std / sample_std, low), high)
        return (values - sample_mean) * ratio + target_mean

    @staticmethod
    def _blendHue(base: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
        delta: np.ndarray = ((target - base + 180.0) % 360.0) - 180.0   # shortest circular step
        return (base + weight * delta) % 360.0

    def _handlePil(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return img
        rgb: np.ndarray = np.asarray(img.convert("RGB"))
        hsv: np.ndarray = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
        hue: np.ndarray = hsv[..., 0] * 2.0
        saturation: np.ndarray = hsv[..., 1] / 255.0
        value: np.ndarray = hsv[..., 2] / 255.0

        achromatic: np.ndarray = saturation < self.saturation_threshold
        brown_mask: np.ndarray = (~achromatic) & self._inBand(hue, *REMAP_BROWN_HUE) \
            & (value < REMAP_BROWN_V)
        # Shadows are excluded from the mask, not just from the blend weight, so the S/V
        # moment-matching statistics are taken from the lit part of the region only.
        lit: np.ndarray = value >= self.shadow_value
        gate: np.ndarray = self._protectionGate(saturation, value)

        out_hue: np.ndarray = hue.copy()
        out_saturation: np.ndarray = saturation.copy()
        out_value: np.ndarray = value.copy()
        for source_color, target in self.correspondence.items():
            if source_color in ACHROMATIC_COLORS or target.target_color == source_color:
                continue
            mask: np.ndarray = self._sourceMask(source_color, hue, achromatic, brown_mask) & lit
            if not mask.any():
                continue
            band: np.ndarray = np.where(mask, self._bandWeight(source_color, hue), 0.0)
            weight: np.ndarray = self._smoothWeight(band) * gate

            matched_saturation: np.ndarray = np.clip(self._matchMoments(
                saturation, saturation[mask], target.saturation_mean, target.saturation_std), 0.0, 1.0)
            matched_value: np.ndarray = np.clip(self._matchMoments(
                value, value[mask], target.value_mean, target.value_std), 0.0, 1.0)

            inverse_weight: np.ndarray = 1.0 - weight
            out_saturation = out_saturation * inverse_weight + matched_saturation * weight
            out_value = out_value * inverse_weight + matched_value * weight
            if target.target_color not in ACHROMATIC_COLORS:       # achromatic target: keep hue, desaturate
                remapped_hue: np.ndarray = self._remapHue(
                    hue, CHROMATIC_HUE_BAND[source_color], CHROMATIC_HUE_BAND[target.target_color])
                hue_delta: np.ndarray = ((remapped_hue - out_hue + 180.0) % 360.0) - 180.0
                out_hue = (out_hue + weight * hue_delta) % 360.0

        hsv_out: np.ndarray = np.stack(
            [(out_hue / 2.0) % 180.0, out_saturation * 255.0, out_value * 255.0], axis=-1)
        rgb_out: np.ndarray = cv2.cvtColor(np.clip(hsv_out, 0.0, 255.0).astype(np.uint8),
                                           cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb_out, mode="RGB")
