import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod


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
