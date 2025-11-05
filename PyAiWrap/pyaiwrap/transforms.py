import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


class ToGrayscale:
    """Convert image to grayscale"""

    def __init__(self, num_output_channels: int = 1):
        """
        Initialize grayscale transform.

        Args:
            num_output_channels: Number of output channels (1 or 3)
                                1: Single channel grayscale
                                3: 3-channel grayscale (same values repeated)
        """
        if num_output_channels not in [1, 3]:
            raise ValueError("num_output_channels must be 1 or 3")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Apply grayscale conversion.

        Args:
            img: PIL Image, Tensor (C, H, W), or numpy array (H, W, C) or (H, W)

        Returns:
            Grayscale image in the same format as input
        """
        if isinstance(img, torch.Tensor):
            # If tensor, convert to grayscale
            if img.shape[0] == 1:
                # Already grayscale
                if self.num_output_channels == 3:
                    return img.repeat(3, 1, 1)
                return img

            # Convert RGB to grayscale using standard weights
            # Y = 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            gray = gray.unsqueeze(0)  # Add channel dimension

            if self.num_output_channels == 3:
                return gray.repeat(3, 1, 1)
            return gray

        elif isinstance(img, np.ndarray):
            # Handle numpy array (H, W, C) or (H, W)
            if img.ndim == 2:
                # Already grayscale (H, W)
                if self.num_output_channels == 3:
                    return np.stack([img] * 3, axis=2)
                return img

            elif img.ndim == 3:
                if img.shape[2] == 1:
                    # Single channel (H, W, 1)
                    if self.num_output_channels == 3:
                        return np.repeat(img, 3, axis=2)
                    return img.squeeze(2)  # Return (H, W)

                # RGB to grayscale (H, W, C) -> (H, W) or (H, W, 3)
                gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

                if self.num_output_channels == 1:
                    return gray  # (H, W)
                else:
                    return np.stack([gray] * 3, axis=2)  # (H, W, 3)
            else:
                raise ValueError(f"Unsupported numpy array shape: {img.shape}")

        elif isinstance(img, Image.Image):
            # If PIL Image
            gray_pil = TF.to_grayscale(img, num_output_channels=self.num_output_channels)
            return gray_pil

        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ExtractRedChannelTo3Channel:
    """Extract red channel and create 3-channel image with only red populated (zeros in green and blue)"""

    def __call__(self, img):
        """
        Extract red channel and create RGB image with only red channel having values.

        Args:
            img: PIL Image or Tensor

        Returns:
            Tensor of shape (3, H, W) with format [red_values, 0, 0]
        """
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:
                # Extract red channel
                red = img[0:1, :, :]
                result = torch.zeros_like(img)
                result[0:1, :, :] = red
                return result
            else:
                # Single channel, put it in red
                result = torch.zeros(3, img.shape[1], img.shape[2])
                result[0:1, :, :] = img
                return result
        else:
            # PIL Image
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            red_channel = img_array[:, :, 0]

            # Create 3-channel array with only red
            result = np.zeros_like(img_array)
            result[:, :, 0] = red_channel

            # Convert to tensor
            result = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
            return result

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExtractGreenChannelTo3Channel:
    """Extract green channel and create 3-channel image with only green populated (zeros in red and blue)"""

    def __call__(self, img):
        """
        Extract green channel and create RGB image with only green channel having values.

        Args:
            img: PIL Image or Tensor

        Returns:
            Tensor of shape (3, H, W) with format [0, green_values, 0]
        """
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:
                # Extract green channel
                green = img[1:2, :, :]
                result = torch.zeros_like(img)
                result[1:2, :, :] = green
                return result
            else:
                # Single channel, put it in green
                result = torch.zeros(3, img.shape[1], img.shape[2])
                result[1:2, :, :] = img
                return result
        else:
            # PIL Image
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            green_channel = img_array[:, :, 1]

            # Create 3-channel array with only green
            result = np.zeros_like(img_array)
            result[:, :, 1] = green_channel

            # Convert to tensor
            result = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
            return result

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExtractBlueChannelTo3Channel:
    """Extract blue channel and create 3-channel image with only blue populated (zeros in red and green)"""

    def __call__(self, img):
        """
        Extract blue channel and create RGB image with only blue channel having values.

        Args:
            img: PIL Image or Tensor

        Returns:
            Tensor of shape (3, H, W) with format [0, 0, blue_values]
        """
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 3:
                # Extract blue channel
                blue = img[2:3, :, :]
                result = torch.zeros_like(img)
                result[2:3, :, :] = blue
                return result
            else:
                # Single channel, put it in blue
                result = torch.zeros(3, img.shape[1], img.shape[2])
                result[2:3, :, :] = img
                return result
        else:
            # PIL Image
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            blue_channel = img_array[:, :, 2]

            # Create 3-channel array with only blue
            result = np.zeros_like(img_array)
            result[:, :, 2] = blue_channel

            # Convert to tensor
            result = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
            return result

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExtractGreenChannel:
    """Extract green channel from RGB image"""

    def __init__(self, num_output_channels: int = 1):
        """
        Initialize green channel extraction.

        Args:
            num_output_channels: Number of output channels (1 or 3)
                                1: Single green channel
                                3: Green channel repeated 3 times
        """
        if num_output_channels not in [1, 3]:
            raise ValueError("num_output_channels must be 1 or 3")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Extract green channel.

        Args:
            img: PIL Image, Tensor (C, H, W), or numpy array (H, W, C)

        Returns:
            Green channel in the same format as input
        """
        if isinstance(img, torch.Tensor):
            # If tensor, extract green channel (index 1)
            if img.shape[0] == 1:
                # Already single channel, just return it
                if self.num_output_channels == 3:
                    return img.repeat(3, 1, 1)
                return img

            green = img[1:2, :, :]  # Extract green channel, keep dimension

            if self.num_output_channels == 3:
                return green.repeat(3, 1, 1)
            return green

        elif isinstance(img, np.ndarray):
            # Handle numpy array (H, W, C)
            if img.ndim == 2:
                # Already single channel
                if self.num_output_channels == 3:
                    return np.stack([img] * 3, axis=2)
                return img

            elif img.ndim == 3:
                if img.shape[2] == 1:
                    # Single channel (H, W, 1)
                    if self.num_output_channels == 3:
                        return np.repeat(img, 3, axis=2)
                    return img.squeeze(2)

                # Extract green channel (index 1)
                green = img[:, :, 1]

                if self.num_output_channels == 1:
                    return green  # (H, W)
                else:
                    return np.stack([green] * 3, axis=2)  # (H, W, 3)
            else:
                raise ValueError(f"Unsupported numpy array shape: {img.shape}")

        elif isinstance(img, Image.Image):
            # If PIL Image, convert to RGB first
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Convert to numpy array
            img_array = np.array(img)
            green_channel = img_array[:, :, 1]  # Extract green channel

            if self.num_output_channels == 1:
                # Return as single channel PIL Image
                return Image.fromarray(green_channel, mode='L')
            else:
                # Return as 3-channel PIL Image
                green_3ch = np.stack([green_channel] * 3, axis=2)
                return Image.fromarray(green_3ch.astype(np.uint8), mode='RGB')

        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ExtractRedChannel:
    """Extract red channel from RGB image"""

    def __init__(self, num_output_channels: int = 1):
        """
        Initialize red channel extraction.

        Args:
            num_output_channels: Number of output channels (1 or 3)
        """
        if num_output_channels not in [1, 3]:
            raise ValueError("num_output_channels must be 1 or 3")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """Extract red channel"""
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1:
                if self.num_output_channels == 3:
                    return img.repeat(3, 1, 1)
                return img

            red = img[0:1, :, :]

            if self.num_output_channels == 3:
                return red.repeat(3, 1, 1)
            return red

        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                if self.num_output_channels == 3:
                    return np.stack([img] * 3, axis=2)
                return img

            elif img.ndim == 3:
                if img.shape[2] == 1:
                    if self.num_output_channels == 3:
                        return np.repeat(img, 3, axis=2)
                    return img.squeeze(2)

                red = img[:, :, 0]

                if self.num_output_channels == 1:
                    return red
                else:
                    return np.stack([red] * 3, axis=2)
            else:
                raise ValueError(f"Unsupported numpy array shape: {img.shape}")

        elif isinstance(img, Image.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            red_channel = img_array[:, :, 0]

            if self.num_output_channels == 1:
                return Image.fromarray(red_channel, mode='L')
            else:
                red_3ch = np.stack([red_channel] * 3, axis=2)
                return Image.fromarray(red_3ch.astype(np.uint8), mode='RGB')

        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ExtractBlueChannel:
    """Extract blue channel from RGB image"""

    def __init__(self, num_output_channels: int = 1):
        """
        Initialize blue channel extraction.

        Args:
            num_output_channels: Number of output channels (1 or 3)
        """
        if num_output_channels not in [1, 3]:
            raise ValueError("num_output_channels must be 1 or 3")
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """Extract blue channel"""
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1:
                if self.num_output_channels == 3:
                    return img.repeat(3, 1, 1)
                return img

            blue = img[2:3, :, :]

            if self.num_output_channels == 3:
                return blue.repeat(3, 1, 1)
            return blue

        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                if self.num_output_channels == 3:
                    return np.stack([img] * 3, axis=2)
                return img

            elif img.ndim == 3:
                if img.shape[2] == 1:
                    if self.num_output_channels == 3:
                        return np.repeat(img, 3, axis=2)
                    return img.squeeze(2)

                blue = img[:, :, 2]

                if self.num_output_channels == 1:
                    return blue
                else:
                    return np.stack([blue] * 3, axis=2)
            else:
                raise ValueError(f"Unsupported numpy array shape: {img.shape}")

        elif isinstance(img, Image.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            blue_channel = img_array[:, :, 2]

            if self.num_output_channels == 1:
                return Image.fromarray(blue_channel, mode='L')
            else:
                blue_3ch = np.stack([blue_channel] * 3, axis=2)
                return Image.fromarray(blue_3ch.astype(np.uint8), mode='RGB')

        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def __repr__(self):
        return f"{self.__class__.__name__}(num_output_channels={self.num_output_channels})"


class ExtractChannel:
    """Extract specific channel from image by index"""

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
            if img.shape[0] == 1:
                if self.num_output_channels == 3:
                    return img.repeat(3, 1, 1)
                return img

            channel = img[self.channel_index:self.channel_index+1, :, :]

            if self.num_output_channels == 3:
                return channel.repeat(3, 1, 1)
            return channel

        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                if self.num_output_channels == 3:
                    return np.stack([img] * 3, axis=2)
                return img

            elif img.ndim == 3:
                if img.shape[2] == 1:
                    if self.num_output_channels == 3:
                        return np.repeat(img, 3, axis=2)
                    return img.squeeze(2)

                channel = img[:, :, self.channel_index]

                if self.num_output_channels == 1:
                    return channel
                else:
                    return np.stack([channel] * 3, axis=2)
            else:
                raise ValueError(f"Unsupported numpy array shape: {img.shape}")

        elif isinstance(img, Image.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            channel = img_array[:, :, self.channel_index]

            if self.num_output_channels == 1:
                return Image.fromarray(channel, mode='L')
            else:
                channel_3ch = np.stack([channel] * 3, axis=2)
                return Image.fromarray(channel_3ch.astype(np.uint8), mode='RGB')

        else:
            raise TypeError(f"Unsupported type: {type(img)}")

    def __repr__(self):
        channel_names = ['Red', 'Green', 'Blue']
        return (f"{self.__class__.__name__}(channel={channel_names[self.channel_index]}, "
                f"num_output_channels={self.num_output_channels})")
