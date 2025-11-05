import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch
import random
from typing import List, Tuple


class PairedImageFolder(Dataset):
    def __init__(self, images_folder_path, modification_transform, resize_transform):
        """
        images_folder_path: path to folder with images
        modification_transform: transform applied to modified images
        resize_transform: transform applied to real images
        """

        self.modified_dataset = ImageFolder(images_folder_path, transform=modification_transform)
        self.real_dataset = ImageFolder(images_folder_path, transform=resize_transform)

        assert len(self.modified_dataset) == len(self.real_dataset), \
            f"Dataset length mismatch: {len(self.modified_dataset)} vs {len(self.real_dataset)}"

        # Verify that pairs correspond
        self._verify_pairs()

    def _verify_pairs(self):
        """Verify that modified_dataset[i] and real_dataset[i] are the same image"""

        for iterator in range(len(self.modified_dataset)):
            mod_path, _ = self.modified_dataset.samples[iterator]
            real_path, _ = self.real_dataset.samples[iterator]

            mod_filename = os.path.basename(mod_path)
            real_filename = os.path.basename(real_path)

            if not mod_filename == real_filename:
                raise ValueError(
                    f"Pair mismatch at index {iterator}:\n"
                    f"  Modified: {mod_path}\n"
                    f"  Real:     {real_path}"
                )

    def __len__(self):
        return len(self.modified_dataset)

    def __getitem__(self, idx):
        modified_img, modified_label = self.modified_dataset[idx]
        real_img, real_label = self.real_dataset[idx]

        return modified_img, real_img, modified_label, real_label


class RandomInpaintingMask:
    def __init__(self, mask_sizes: List[int] = [3, 32],
                 num_masks_range: Tuple[int, int] | List[int] = (1, 5)):
        self.mask_sizes = mask_sizes
        self.num_masks_range = num_masks_range

    def __call__(self, image: torch.Tensor, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, H, W = image.shape
        mask = torch.ones(1, H, W)
        random.seed(seed)
        num_masks = random.randint(*self.num_masks_range)

        for _ in range(num_masks):
            mask_size = random.choice(self.mask_sizes)

            if mask_size >= min(H, W):
                mask.fill_(0)
                continue

            x = random.randint(0, W - mask_size)
            y = random.randint(0, H - mask_size)
            mask[:, y:y+mask_size, x:x+mask_size] = 0
        random.seed(None)
        masked_image = image * mask
        return masked_image, mask


class InpaintingImageFolder(Dataset):
    """Image inpainting dataset using PairedImageFolder"""
    def __init__(
        self,
        root_dir: str,
        resize_transform=None,
        mask_sizes: List[int] = [3, 32],
        num_masks_range: Tuple[int, int] = (1, 5)
    ):
        self._base_dataset = ImageFolder(root=root_dir, transform=resize_transform)

        self._mask_transform = RandomInpaintingMask(mask_sizes, num_masks_range)

    def __len__(self):
        return len(self._base_dataset)

    def __getitem__(self, idx):
        real_image, real_label = self._base_dataset[idx]

        masked_image, mask = self._mask_transform(real_image, idx)

        return masked_image, real_image, mask, real_label
