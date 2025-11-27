import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch
import random
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import pickle
from scipy.ndimage import zoom


class PairedImageFolder(Dataset):
    def __init__(self, images_folder_path, input_transform, target_transform):
        """
        images_folder_path: path to folder with images
        modification_transform: transform applied to modified images
        resize_transform: transform applied to real images
        """

        self._input_dataset = ImageFolder(images_folder_path, transform=input_transform)
        self._target_dataset = ImageFolder(images_folder_path, transform=target_transform)

        assert len(self._input_dataset) == len(self._target_dataset), \
            f"Dataset length mismatch: {len(self._input_dataset)} vs {len(self._target_dataset)}"

        # Verify that pairs correspond
        self._verify_pairs()

    def _verify_pairs(self):
        """Verify that modified_dataset[i] and real_dataset[i] are the same image"""

        for iterator in range(len(self._input_dataset)):
            mod_path, _ = self._input_dataset.samples[iterator]
            real_path, _ = self._target_dataset.samples[iterator]

            mod_filename = os.path.basename(mod_path)
            real_filename = os.path.basename(real_path)

            if not mod_filename == real_filename:
                raise ValueError(
                    f"Pair mismatch at index {iterator}:\n"
                    f"  Modified: {mod_path}\n"
                    f"  Real:     {real_path}"
                )

    def __len__(self):
        return len(self._input_dataset)

    def __getitem__(self, idx):
        modified_img, modified_label = self._input_dataset[idx]
        real_img, real_label = self._target_dataset[idx]

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


class KneeMRISegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        target_size: Optional[Tuple[int, int, int]] = None
    ):
        self.data_root = data_root
        self.metadata_path = metadata_path
        self.target_size = target_size

        self.metadata = pd.read_csv(metadata_path)

        self.valid_samples = []
        for idx, row in self.metadata.iterrows():
            filename = row['volumeFilename']
            if self.findVolumePath(filename):
                self.valid_samples.append(idx)

    def findVolumePath(self, filename: str) -> Optional[str]:
        for vol_num in range(1, 11):
            vol_dir = f"vol{vol_num:02d}"
            file_path = os.path.join(self.data_root, vol_dir, filename)
            if os.path.exists(file_path):
                return file_path
        return None

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx = self.valid_samples[idx]
        row = self.metadata.iloc[sample_idx]

        filename = row['volumeFilename']
        volume_path = self.findVolumePath(filename)
        volume = self.loadVolume(volume_path)

        seg_mask = self.createSegmentationMask(volume.shape, row)

        if self.target_size:
            volume, seg_mask = self.resizeVolumeAndMask(volume, seg_mask, self.target_size)

        volume = self.preprocessVolume(volume)

        # [1, D, H, W]
        volume_tensor = torch.from_numpy(volume.copy()).float().unsqueeze(0)
        # [1, D, H, W]
        seg_mask_tensor = torch.from_numpy(seg_mask.copy()).long().unsqueeze(0)

        return volume_tensor, seg_mask_tensor

    def loadVolume(self, file_path: str) -> np.ndarray:
        with open(file_path, 'rb') as f:
            volume = pickle.load(f)
        return volume

    def createSegmentationMask(self, vol_shape: Tuple, row: pd.Series) -> np.ndarray:
        # [D, H, W]
        seg_mask = np.zeros(vol_shape, dtype=np.uint8)

        x, y, z = row['roiX'], row['roiY'], row['roiZ']
        w, h, d = row['roiWidth'], row['roiHeight'], row['roiDepth']

        # Convert diagnosis to class: 0->1, 1->2, 2->3
        acl_class = row['aclDiagnosis'] + 1

        z_start, z_end = max(0, z), min(vol_shape[0], z + d)
        y_start, y_end = max(0, y), min(vol_shape[1], y + h)
        x_start, x_end = max(0, x), min(vol_shape[2], x + w)

        seg_mask[z_start:z_end, y_start:y_end, x_start:x_end] = acl_class

        return seg_mask

    def preprocessVolume(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        return volume

    def resizeVolumeAndMask(self, volume: np.ndarray, seg_mask: np.ndarray, target_size: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:

        current_depth, current_height, current_width = volume.shape
        target_depth, target_height, target_width = target_size

        depth_factor = target_depth / current_depth
        height_factor = target_height / current_height
        width_factor = target_width / current_width

        resized_volume = zoom(volume, (depth_factor, height_factor, width_factor), order=1)

        resized_mask = zoom(seg_mask, (depth_factor, height_factor, width_factor), order=0)

        return resized_volume, resized_mask
