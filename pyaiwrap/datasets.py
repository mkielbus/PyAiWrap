import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import torch
import random
from typing import Callable, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import pickle
from scipy.ndimage import zoom

from pyaiwrap.segmentation_masks import SegmentationMaskEncoder, loadLabelMap
from pyaiwrap.transforms import PairedGeometricAugmentation, PathAwareImageTransform


class EdgesDataset(Dataset):
    def __init__(self, edges_filepaths: List[str], resize_transform: Optional[transforms.Resize]):
        """
        edges_filepaths: paths to single-channel edge images, ordered to match the paired image dataset
        resize_transform: the resize applied to the paired input images, so edges stay aligned with them
        """

        self._edges_filepaths = edges_filepaths
        edge_transforms = [resize_transform] if resize_transform is not None else []
        edge_transforms += [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        self._transform = transforms.Compose(edge_transforms)

    def __len__(self):
        return len(self._edges_filepaths)

    def __getitem__(self, idx):
        edges_img = Image.open(self._edges_filepaths[idx])
        return self._transform(edges_img)


class PairedImageFolder(Dataset):
    def __init__(self, images_folder_path: str,
                 input_transform: Callable,
                 target_transform: Callable,
                 segmentation_pairing: Optional[str] = None,
                 shared_augmentation: Optional[Callable] = None,
                 target_augmentation: Optional[Callable] = None,
                 input_augmentation: Optional[Callable] = None,
                 mask_folder_path: Optional[str] = None,
                 mask_encoder: Optional[SegmentationMaskEncoder] = None,
                 image_size: Optional[int] = None) -> None:
        """
        images_folder_path: path to folder with images
        input_transform: transform producing the model input from a PIL image
        target_transform: transform producing the target from the same PIL image
        segmentation_pairing: optional path to a csv with header image_filepath,edges_filepath
            pairing each image with its edges image; when given, edges are stacked onto the
            input image channels
        shared_augmentation: optional callable applied once to each PIL image before the
            input/target transforms, so both branches receive the identical geometric view
            (train-time flip/crop). Sampled once per __getitem__. Because the edges path is
            not co-augmented, it cannot be combined with segmentation_pairing (raises ValueError).
        target_augmentation: optional callable applied to the target branch only, after the
            shared geometric augmentation (train-time photometric augmentation, e.g. chroma
            jitter). Keeps the input pristine, so recomputed luminance is unaffected.
        input_augmentation: optional callable applied to the input branch only, after the
            shared geometric augmentation (train-time tone jitter). The mirror image of
            target_augmentation: the target keeps its true colours while the input's luminance
            is perturbed, which regularises without changing what is being learnt. The two are
            sampled independently, so a sample may receive either, both or neither.
        mask_folder_path: optional folder of SAM label maps (one uint8 PNG per image, named
            after the image's stem). The map is co-augmented with the image -- same flip, same
            crop box, NEAREST resampling -- then encoded by `mask_encoder` and stacked onto
            the input's channels, behind the channels the pretrained extractors consume. The
            merge network splits them off again (ConvAttenColorizationNetwork._separateAuxiliaryChannels),
            so the frozen R/G/B extractors keep seeing luminance alone and only the trainable
            UNet is conditioned on the segmentation.
        mask_encoder: how those label maps become channels; defaults to boundary + area.
        image_size: side the masks are resampled to when there is no augmentation to do it
            (validation). Inferred from the augmentation or from the input transform's Resize
            when not given.

        The image is decoded once (single ImageFolder with transform=None), which removes the
        previous double-decode and makes the input/target pairing correct by construction.
        """

        if segmentation_pairing is not None and shared_augmentation is not None:
            raise ValueError(
                "shared_augmentation cannot be combined with segmentation_pairing: edge maps "
                "are not co-augmented and would misalign with the augmented image."
            )
        if segmentation_pairing is not None and mask_folder_path is not None:
            raise ValueError(
                "segmentation_pairing and mask_folder_path both stack channels onto the input; "
                "pick one."
            )
        if mask_folder_path is not None and shared_augmentation is not None \
                and not isinstance(shared_augmentation, PairedGeometricAugmentation):
            raise ValueError(
                "mask_folder_path with augmentation requires a PairedGeometricAugmentation "
                "(see createPairedGeometricAugmentation): the composed transform samples its "
                "crop internally, so the mask cannot be given the same geometry and would be "
                "misaligned with the image."
            )

        self._dataset: ImageFolder = ImageFolder(images_folder_path, transform=None)
        self._input_transform: Callable = input_transform
        self._target_transform: Callable = target_transform
        self._shared_augmentation: Optional[Callable] = shared_augmentation
        self._target_augmentation: Optional[Callable] = target_augmentation
        self._input_augmentation: Optional[Callable] = input_augmentation

        self._edges_dataset: Optional[EdgesDataset] = None
        if segmentation_pairing is not None:
            self._edges_dataset = self._buildEdgesDataset(segmentation_pairing, input_transform)

        self._mask_filepaths: Optional[List[str]] = None
        self._mask_encoder: Optional[SegmentationMaskEncoder] = None
        self._mask_image_size: Optional[int] = None
        if mask_folder_path is not None:
            self._mask_filepaths = self._buildMaskFilepaths(mask_folder_path)
            self._mask_encoder = mask_encoder or SegmentationMaskEncoder()
            self._mask_image_size = self._resolveMaskImageSize(image_size, input_transform)

    def _buildEdgesDataset(self, segmentation_pairing: str, input_transform: Callable) -> EdgesDataset:
        """Build an edges dataset ordered like the base dataset from the pairing csv"""

        pairing: pd.DataFrame = pd.read_csv(segmentation_pairing)
        edges_by_filename: Dict[str, str] = {
            os.path.basename(image_path): edges_path
            for image_path, edges_path in zip(pairing['image_filepath'], pairing['edges_filepath'])
        }

        edges_filepaths: List[str] = []
        for image_path, _ in self._dataset.samples:
            filename: str = os.path.basename(image_path)
            if filename not in edges_by_filename:
                raise ValueError(
                    f"No edges entry in {segmentation_pairing} for image {image_path}"
                )
            edges_filepaths.append(edges_by_filename[filename])

        return EdgesDataset(edges_filepaths, self._findResizeTransform(input_transform))

    def _buildMaskFilepaths(self, mask_folder_path: str) -> List[str]:
        """Pair every image with its label map by filename stem, in dataset order.

        A missing map is an error rather than a silently unconditioned sample: the model has
        the channels wired into enc1, so a gap would have to be filled with something, and
        every choice of filler teaches the network that "no segmentation" is a state the
        world can be in.
        """
        mask_filepaths: List[str] = []
        for image_path, _label in self._dataset.samples:
            stem: str = os.path.splitext(os.path.basename(image_path))[0]
            mask_path: str = os.path.join(mask_folder_path, f"{stem}.png")
            if not os.path.isfile(mask_path):
                raise ValueError(f"No label map {mask_path} for image {image_path}")
            mask_filepaths.append(mask_path)
        return mask_filepaths

    def _resolveMaskImageSize(self, image_size: Optional[int], input_transform: Callable) -> int:
        """The side a label map is resampled to when no augmentation is doing the resampling."""
        if image_size is not None:
            return image_size
        if isinstance(self._shared_augmentation, PairedGeometricAugmentation):
            return self._shared_augmentation.image_size

        resize: Optional[transforms.Resize] = self._findResizeTransform(input_transform)
        if resize is None:
            raise ValueError(
                "mask_folder_path needs image_size: it could not be inferred from the "
                "augmentation or from a Resize in the input transform."
            )
        size = resize.size
        return size if isinstance(size, int) else size[0]

    @staticmethod
    def _findResizeTransform(input_transform: Callable) -> Optional[transforms.Resize]:
        """Extract the resize step from the input transform so edges are resized the same way"""

        if isinstance(input_transform, transforms.Resize):
            return input_transform
        if isinstance(input_transform, transforms.Compose):
            for transform in input_transform.transforms:
                if isinstance(transform, transforms.Resize):
                    return transform
        return None

    def __len__(self) -> int:
        return len(self._dataset)

    def _applyTargetAugmentation(self, image: Image.Image, idx: int) -> Image.Image:
        """Path-aware augmentations (e.g. the cluster-version remap, whose correspondence depends
        on the source image's cluster) are told which file the image came from; plain photometric
        augmentations are called with the image alone."""
        if isinstance(self._target_augmentation, PathAwareImageTransform):
            return self._target_augmentation(image, self._dataset.samples[idx][0])
        return self._target_augmentation(image)

    def _loadMask(self, idx: int) -> Image.Image:
        return loadLabelMap(self._mask_filepaths[idx])

    def _encodeMask(self, mask: Image.Image) -> torch.Tensor:
        return self._mask_encoder(np.asarray(mask, dtype=np.uint8))

    def _resizeMask(self, mask: Image.Image) -> Image.Image:
        """Validation's square-off, mirroring what the input transform's Resize does."""
        size: int = self._mask_image_size
        if mask.size == (size, size):
            return mask
        return mask.resize((size, size), Image.NEAREST)

    def _applySharedAugmentation(self, image: Image.Image, mask: Optional[Image.Image]
                                 ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """One geometric draw for the image, the target, and the label map alike."""
        if not isinstance(self._shared_augmentation, PairedGeometricAugmentation):
            return self._shared_augmentation(image), mask

        width, height = image.size
        parameters = self._shared_augmentation.sampleParameters(width, height)
        image = self._shared_augmentation.applyToImage(image, parameters)
        if mask is not None:
            mask = self._shared_augmentation.applyToMask(mask, parameters)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        image, label = self._dataset[idx]           # PIL RGB image, decoded once
        mask: Optional[Image.Image] = self._loadMask(idx) if self._mask_filepaths is not None else None

        if self._shared_augmentation is not None:
            # single geometric op shared by both branches, and by the label map when there is one
            image, mask = self._applySharedAugmentation(image, mask)
        elif mask is not None:
            mask = self._resizeMask(mask)

        target_image = image
        if self._target_augmentation is not None:
            target_image = self._applyTargetAugmentation(image, idx)  # target-side aug only

        input_image = image
        if self._input_augmentation is not None:
            input_image = self._input_augmentation(image)  # input-side aug only

        model_input: torch.Tensor = self._input_transform(input_image)
        target: torch.Tensor = self._target_transform(target_image)

        if self._edges_dataset is not None:
            edges_img: torch.Tensor = self._edges_dataset[idx]
            model_input = torch.cat([model_input, edges_img], dim=0)

        if mask is not None:
            model_input = torch.cat([model_input, self._encodeMask(mask)], dim=0)

        return model_input, target, label, label


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
