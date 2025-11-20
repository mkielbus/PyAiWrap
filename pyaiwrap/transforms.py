import torch
import torchvision.transforms.functional as TF
import kornia
from torchvision import transforms
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
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
    # Ensure proper dimensions
    if lChannel.dim() == 3:
        lChannel = lChannel.unsqueeze(0)
    if abChannels.dim() == 3:
        abChannels = abChannels.unsqueeze(0)

    # Combine L and AB channels - both in Kornia's native ranges
    lab = torch.cat([lChannel, abChannels], dim=1)

    rgb = kornia.color.lab_to_rgb(lab)

    # Remove batch dimension if needed
    if rgb.shape[0] == 1:
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
    if labTensor.dim() == 3:
        labTensor = labTensor.unsqueeze(0)

    rgb = kornia.color.lab_to_rgb(labTensor)

    if rgb.shape[0] == 1:
        rgb = rgb.squeeze(0)

    return rgb


class ChannelType(Enum):
    """Enum representing different channel types."""
    RGB = "RGB"
    LAB = "LAB"
    AB = "AB"
    AB_TO_3CH = "ab_to_3ch"
    LUMINANCE = "luminance"
    R = "R"
    G = "G"
    B = "B"


class TransformFactory(ABC):
    """Abstract factory interface for creating transform compositions."""

    @abstractmethod
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        pass


class RGBTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])


class LABTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB()
        ])


class ABTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        if output_channels not in [2, 3]:
            raise ValueError("output_channels must be 2 or 3 for 'ab' channel_type")
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB(),
            ExtractABChannels(num_output_channels=output_channels)
        ])


class ABTo3ChannelTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RGBToLAB(),
            ExtractABChannelsTo3Channel()
        ])


class LuminanceTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        if not is_input:
            raise ValueError("luminance can only be used for input channels, not target channels")
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ToGrayscale(num_output_channels=output_channels),
            transforms.ToTensor()
        ])


class RedChannelTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractRedChannel(num_output_channels=output_channels)
        ])


class GreenChannelTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractGreenChannel(num_output_channels=output_channels)
        ])


class BlueChannelTransformFactory(TransformFactory):
    def createTransform(self, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ExtractBlueChannel(num_output_channels=output_channels)
        ])


class ChannelTransformCreator:
    """Main creator class that uses the abstract factory pattern."""

    _factories: Dict[ChannelType, TransformFactory] = {
        ChannelType.RGB: RGBTransformFactory(),
        ChannelType.LAB: LABTransformFactory(),
        ChannelType.AB: ABTransformFactory(),
        ChannelType.AB_TO_3CH: ABTo3ChannelTransformFactory(),
        ChannelType.LUMINANCE: LuminanceTransformFactory(),
        ChannelType.R: RedChannelTransformFactory(),
        ChannelType.G: GreenChannelTransformFactory(),
        ChannelType.B: BlueChannelTransformFactory()
    }

    @classmethod
    def getTransform(cls, channel_type: str, image_size: int, output_channels: int, is_input: bool = True) -> transforms.Compose:
        """Get the appropriate transform for the given channel type."""
        try:
            channel_enum = ChannelType(channel_type)
            factory = cls._factories[channel_enum]
            return factory.createTransform(image_size, output_channels, is_input)
        except (KeyError, ValueError):
            raise ValueError(f"channel_type must be 'luminance', 'R', 'G', 'B', or 'RGB', got '{channel_type}'")
