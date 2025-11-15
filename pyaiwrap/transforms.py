import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
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


class RGBToLab(ImageTransform):
    """Convert RGB image to LAB color space and return as tensor"""

    def __init__(self):
        """Initialize RGB to LAB conversion"""

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
        """Convert tensor from RGB to LAB"""
        if img.shape[0] != 3:
            raise ValueError("Input tensor must have 3 channels for RGB to LAB conversion")

        # Assuming tensor is in range [0, 1] and shape [C, H, W]
        rgb = img.permute(1, 2, 0).numpy()  # Convert to HWC for conversion
        lab = self._rgb_to_lab(rgb)
        return torch.from_numpy(lab).permute(2, 0, 1).float()  # Convert back to CHW and ensure float

    def _handleNumpy(self, img):
        """Convert numpy array from RGB to LAB and return as tensor"""
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Input array must have 3 channels for RGB to LAB conversion")

        lab = self._rgb_to_lab(img)
        return torch.from_numpy(lab).permute(2, 0, 1).float()

    def _handlePil(self, img):
        """Convert PIL Image from RGB to LAB and return as tensor"""
        if img.mode != 'RGB':
            img = img.convert('RGB')

        rgb_array = np.array(img)
        lab_array = self._rgb_to_lab(rgb_array)
        return torch.from_numpy(lab_array).permute(2, 0, 1).float()

    def _rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space - returns float array"""
        rgb = rgb.astype(np.float32) / 255.0

        mask = rgb > 0.04045
        rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
        rgb[~mask] = rgb[~mask] / 12.92

        xyz = np.dot(rgb, self._get_rgb_to_xyz_matrix().T)

        xyz = xyz / self._get_reference_white()

        mask = xyz > 0.008856
        xyz[mask] = xyz[mask] ** (1/3)
        xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)

        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        L = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)

        L = np.clip(L, 0, 100)
        a = np.clip(a + 128, 0, 255)
        b = np.clip(b + 128, 0, 255)

        lab = np.stack([L, a, b], axis=-1).astype(np.float32)
        return lab

    def _get_rgb_to_xyz_matrix(self):
        """Get RGB to XYZ conversion matrix (sRGB, D65)"""
        return np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])

    def _get_reference_white(self):
        """Get reference white point (D65)"""
        return np.array([0.95047, 1.0, 1.08883])

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExtractABChannels(ImageTransform):
    """Extract A and B channels from LAB color space - expects tensor input"""

    def __init__(self, num_output_channels: int = 2):
        """
        Initialize AB channels extraction.

        Args:
            num_output_channels: Number of output channels (2 or 3)
        """
        if num_output_channels not in [2, 3]:
            raise ValueError("num_output_channels must be 2 or 3")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """Extract A and B channels from LAB tensor"""
        if isinstance(img, torch.Tensor):
            return self._handleTensor(img)
        else:
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).permute(2, 0, 1).float()
            return self._handleTensor(img)

    def _handleTensor(self, img):
        if img.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {img.dim()}")

        if img.shape[0] == 1:
            raise ValueError("Cannot extract AB channels from single channel image")

        if img.shape[0] == 2:
            ab_channels = img
        elif img.shape[0] == 3:
            ab_channels = img[1:3, :, :] / 255.0
        else:
            raise ValueError(f"Unsupported tensor shape: {img.shape}")

        if self.num_output_channels == 3:
            zeros = torch.zeros_like(ab_channels[0:1, :, :])
            return torch.cat([zeros, ab_channels], dim=0)
        else:
            return ab_channels

    def _handleNumpy(self, img):
        """Extract AB channels from numpy array - convert to tensor"""
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return self._handleTensor(img_tensor)

    def _handlePil(self, img):
        """Extract AB channels from PIL Image - convert to tensor"""
        img_tensor = transforms.ToTensor()(img)
        return self._handleTensor(img_tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ExtractABChannelsTo3Channel(ImageTransform):
    """Extract A and B channels and create 3-channel LAB tensor with zeros in L channel"""

    def __init__(self):
        """Initialize AB channels extraction to 3-channel LAB format"""

    def __call__(self, img):
        """Extract AB channels and create 3-channel LAB tensor"""
        if isinstance(img, torch.Tensor):
            return self._handleTensor(img)
        else:
            # Convert to tensor if needed
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).permute(2, 0, 1).float()
            return self._handleTensor(img)

    def _handleTensor(self, img):
        """Extract AB channels from tensor and create 3-channel LAB output"""
        if img.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {img.dim()}")

        if img.shape[0] == 1:
            raise ValueError("Cannot extract AB channels from single channel image")

        if img.shape[0] == 2:
            ab_channels = img
        elif img.shape[0] == 3:
            ab_channels = img[1:3, :, :]
        else:
            raise ValueError(f"Unsupported tensor shape: {img.shape}")

        zeros = torch.zeros_like(ab_channels[0:1, :, :])
        return torch.cat([zeros, ab_channels], dim=0)

    def _handleNumpy(self, img):
        """Handle numpy array - convert to tensor"""
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return self._handleTensor(img_tensor)

    def _handlePil(self, img):
        """Handle PIL Image - convert to tensor"""
        img_tensor = transforms.ToTensor()(img)
        return self._handleTensor(img_tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
