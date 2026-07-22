import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from typing import Tuple, Dict, Any, Optional
from .metrics import Metrics
from .transforms import labToRgb, labToRgbForVisualization
from monai.losses import DiceCELoss

# Autocast dtypes selectable from a config. bfloat16 keeps fp32's exponent range, so unlike
# float16 it needs no GradScaler.
_AUTOCAST_DTYPES: Dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class LPIPSLoss(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) loss.
    Better perceptual metric than VGG features.
    """
    def __init__(self, net: str = 'alex', use_gpu: bool = True):
        """
        Args:
            net: Backbone network - 'alex' (AlexNet), 'vgg' (VGG16), or 'squeeze' (SqueezeNet)
                 'alex' is recommended (fastest and most accurate)
            use_gpu: Whether to use GPU
        """
        super().__init__()

        self.lpips_model = lpips.LPIPS(net=net, verbose=False)

        for param in self.lpips_model.parameters():
            param.requires_grad = False

        self.lpips_model.eval()

        if use_gpu and torch.cuda.is_available():
            self.lpips_model = self.lpips_model.cuda()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated: [B, 3, H, W] - generated images (range [-1, 1] or [0, 1])
            target: [B, 3, H, W] - target images
            mask: [B, 1, H, W] - optional mask (1=keep, 0=inpaint)
                  If provided, only compute LPIPS on masked regions

        Returns:
            lpips_loss: Scalar loss value (lower is better)
        """

        lpips_loss = self.lpips_model(generated, target).mean()

        return lpips_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Pixel loss + LPIPS loss
    """
    def __init__(self,
                 pixel_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 lpips_net: str = 'alex'):
        """
        Args:
            pixel_weight: Weight for pixel-wise MSE loss
            lpips_weight: Weight for LPIPS loss
            lpips_net: LPIPS backbone ('alex', 'vgg', or 'squeeze')
        """
        super().__init__()

        self.pixel_weight = pixel_weight
        self.lpips_weight = lpips_weight

        self.pixel_loss = nn.MSELoss()
        self.lpips_loss = LPIPSLoss(net=lpips_net)

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            generated: [B, 3, H, W] - generated images
            target: [B, 3, H, W] - target images

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        pixel_loss = self.pixel_loss(generated, target)

        lpips_loss = self.lpips_loss(generated, target)

        total_loss = (
            self.pixel_weight * pixel_loss +
            self.lpips_weight * lpips_loss
        )

        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'pixel': pixel_loss.item(),
            'lpips': lpips_loss.item()
        }

        return total_loss, loss_dict


# ----------------------------
# Loss Function
# ----------------------------
def generatorLossFunction(model: nn.Module, batch: tuple) -> torch.Tensor:
    """
    Generator loss with LPIPS.

    Args:
        model: The generator model
        batch: Tuple containing (modified_images, real_images, _, _)

    Returns:
        Loss value
    """
    modified_images, real_images, _, _ = batch

    generated = model(modified_images)

    loss_fn = CombinedLoss(pixel_weight=1.0, lpips_weight=1.0, lpips_net='alex')
    total_loss, loss_dict = loss_fn(generated, real_images)

    return total_loss


class InpaintingCombinedLoss(nn.Module):
    """
    Combined loss for inpainting:
    - L1 loss on holes (high priority)
    - L1 loss on valid regions (consistency)
    - LPIPS loss (perceptual quality)
    - Optional: Style loss, TV loss
    """
    def __init__(self,
                 hole_weight: float = 1.0,
                 valid_weight: float = 1.0,
                 lpips_weight: float = 1.0,
                 lpips_net: str = 'alex'):
        """
        Args:
            hole_weight: Weight for L1 loss in holes
            valid_weight: Weight for L1 loss in valid regions
            lpips_weight: Weight for LPIPS perceptual loss
            lpips_net: LPIPS backbone network
        """
        super().__init__()

        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.lpips_weight = lpips_weight

        self.lpips_loss = LPIPSLoss(net=lpips_net)

    def forward(self, generated: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            generated: [B, 3, H, W] - generated images
            target: [B, 3, H, W] - target images
            mask: [B, 1, H, W] - binary mask (1=valid/keep, 0=hole/inpaint)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        hole_mask = 1 - mask  # [B, 1, H, W]

        # Holes
        hole_loss = F.l1_loss(
            generated * hole_mask,
            target * hole_mask,
            reduction='sum'
        ) / (hole_mask.sum() + 1e-8)

        # Rest of the images
        valid_loss = F.l1_loss(
            generated * mask,
            target * mask,
            reduction='sum'
        ) / (mask.sum() + 1e-8)

        lpips_loss = self.lpips_loss(generated, target)

        total_loss = (
            self.hole_weight * hole_loss +
            self.valid_weight * valid_loss +
            self.lpips_weight * lpips_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'hole_l1': hole_loss.item(),
            'valid_l1': valid_loss.item(),
            'lpips': lpips_loss.item(),
        }

        return total_loss, loss_dict


# ----------------------------
# Inpainting Loss Function
# ----------------------------
def inpaintingLossFunction(model: nn.Module, batch: tuple) -> torch.Tensor:
    """
    Better inpainting loss with LPIPS and multi-component loss.

    Args:
        model: The inpainting generator model
        batch: Tuple containing (masked_image, original_image, mask, _)

    Returns:
        Loss value
    """
    masked_image, original_image, mask, _ = batch

    # [B, 4, H, W] = RGB channels + mask channel
    model_input = torch.cat([masked_image, mask], dim=1)
    generated = model(model_input)

    loss_fn = InpaintingCombinedLoss(
        hole_weight=1.0,
        valid_weight=1.0,
        lpips_weight=1.5,
        lpips_net='alex'
    )

    total_loss, loss_dict = loss_fn(generated, original_image, mask)

    return total_loss


class GANLoss:
    """
    GAN loss function that handles both generator and discriminator training.
    """

    def __init__(self, criterion: nn.Module = None):
        """
        Initialize GAN loss function.

        Args:
            criterion: Loss criterion (default: nn.L1Loss)
        """
        self.criterion = criterion if criterion is not None else nn.L1Loss()

    def __call__(
        self,
        models: Dict[str, nn.Module],
        batch: Tuple,
        metrics: Metrics,
        gradient_clip: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute GAN loss for a batch.

        Args:
            models: Dictionary containing 'generator' and 'discriminator'
            batch: Tuple of (modified_images, real_images, ...)
            metrics: Metrics object to accumulate batch statistics

        Returns:
            Dictionary with 'loss' key for backpropagation
        """
        generator = models['generator']
        discriminator = models['discriminator']

        modified_images, real_images = batch[0], batch[1]
        batch_size = real_images.size(0)
        device = real_images.device

        fake_images = generator(modified_images)

        label_real = torch.ones((batch_size, 1), dtype=torch.float, device=device)
        label_fake = torch.zeros((batch_size, 1), dtype=torch.float, device=device)

        output_real = discriminator(real_images)
        loss_real = self.criterion(output_real, label_real)
        acc_real = 1.0 - torch.abs(output_real - label_real).mean().item()

        output_fake_for_disc = discriminator(fake_images.detach())
        loss_fake = self.criterion(output_fake_for_disc, label_fake)
        acc_fake = 1.0 - torch.abs(output_fake_for_disc - label_fake).mean().item()

        loss_discriminator = loss_real + loss_fake

        if torch.is_grad_enabled():
            loss_discriminator.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=gradient_clip)

        output_fake_for_gen = discriminator(fake_images)
        label_gen = torch.ones((batch_size, 1), dtype=torch.float, device=device)
        loss_generator = self.criterion(output_fake_for_gen, label_gen)

        if torch.is_grad_enabled():
            loss_generator.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=gradient_clip)

        total_loss = loss_generator + loss_discriminator

        metrics.accumulate({
            'generator_loss': loss_generator.item(),
            'discriminator_loss': loss_discriminator.item(),
            'discriminator_real_acc': acc_real,
            'discriminator_fake_acc': acc_fake
        })

        return {
            'loss': total_loss,
            'generator_loss': loss_generator,
            'discriminator_loss': loss_discriminator,
            'loss_real': loss_real,
            'loss_fake': loss_fake
        }


class VAELoss:
    """Loss function for Variational Autoencoder"""

    def __init__(
        self,
        reconstruction_loss_fn: nn.Module = nn.MSELoss(reduction='mean'),
        kl_weight: float = 1.0,
    ):
        """
        Initialize VAE loss function.

        Args:
            reconstruction_loss_fn: Loss function for reconstruction (default: MSELoss)
            kl_weight: Weight for KL divergence term (beta in beta-VAE)
        """
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_weight = kl_weight

    def __call__(
        self,
        models: Dict[str, nn.Module],
        batch: Tuple,
        metrics: Metrics,
        gradient_clip: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate VAE loss and update metrics.

        Args:
            models: Dictionary containing the VAE model (e.g., {'vae': vae_model})
            batch: Tuple of (input_data, target_data) or just (input_data,)
            metrics: Metrics object with accumulate() method
            gradient_clip: Gradient clipping value (None to disable)

        Returns:
            Dictionary with 'loss' key containing the total loss tensor
        """

        if 'vae' not in models:
            raise ValueError("models dictionary must contain 'vae' key")
        vae = models['vae']

        input_data = batch[0]
        target_data = batch[1]

        reconstruction, mu, logvar = vae(input_data)

        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, target_data)

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence = kl_divergence / input_data.size(0)

        total_loss = reconstruction_loss + self.kl_weight * kl_divergence

        if torch.is_grad_enabled():
            total_loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=gradient_clip)

        metrics.accumulate({
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_divergence': kl_divergence.item()
        })

        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_divergence': kl_divergence
        }


def calculateColorfulnessLoss(images: torch.Tensor) -> torch.Tensor:
    """
    Calculate colorfulness metric as differentiable loss.
    Based on Hasler & Süsstrunk (2003).

    Args:
        images: RGB images in range [0, 1], shape (B, 3, H, W)

    Returns:
        Mean colorfulness score for the batch
    """
    images_scaled = images * 255.0

    R = images_scaled[:, 0, :, :]
    G = images_scaled[:, 1, :, :]
    B = images_scaled[:, 2, :, :]

    rg = R - G
    yb = 0.5 * (R + G) - B

    sigma_rg = torch.std(rg.reshape(rg.shape[0], -1), dim=1)
    sigma_yb = torch.std(yb.reshape(yb.shape[0], -1), dim=1)

    mu_rg = torch.mean(rg.reshape(rg.shape[0], -1), dim=1)
    mu_yb = torch.mean(yb.reshape(yb.shape[0], -1), dim=1)

    sigma_rgyb = torch.sqrt(sigma_rg**2 + sigma_yb**2)
    mu_rgyb = torch.sqrt(mu_rg**2 + mu_yb**2)

    M = sigma_rgyb + 0.3 * mu_rgyb

    return M.mean()


class GeneratorColorizationLoss:
    """Loss function for image reconstruction generator with colorfulness metric"""

    def __init__(
        self,
        reconstruction_loss_fn: nn.Module = nn.MSELoss(),
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        colorfulness_weight: float = 0.0,
        colorfulness_target: Optional[float] = None,
        use_lpips: bool = False,
        lpips_net: str = 'alex',
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        input_channel: str = "RGB",
        target_channel: str = "RGB",
        mixed_precision: bool = False,
        mixed_precision_dtype: str = "bfloat16"
    ):
        """
        Initialize generator loss function.

        Args:
            reconstruction_loss_fn: Loss function for reconstruction (default: MSELoss)
            recon_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss (0 to disable)
            colorfulness_weight: Weight for colorfulness loss (0 to disable)
            colorfulness_target: Target colorfulness value (if None, matches original)
            use_lpips: Whether to use LPIPS for perceptual loss (default: False)
            lpips_net: Network to use for LPIPS ('alex', 'vgg', 'squeeze') (default: 'alex')
            device: Device to place LPIPS network on
            input_channel: Type of input channel ("RGB", "R", "G", "B", "LAB", "AB", "luminance")
            target_channel: Type of target channel ("RGB", "R", "G", "B", "LAB", "AB",
                            "LAB_A", "LAB_B", "luminance")
            mixed_precision: Run the GENERATOR FORWARD under torch.autocast (default False =
                             unchanged fp32 behaviour). Every loss term stays in fp32: the
                             generator output is cast back before reconstruction/LPIPS/LAB
                             conversion/colourfulness, because labToRgb and the colourfulness
                             statistic are precision-sensitive and LPIPS is a frozen fp32 net.
                             Backward runs outside the autocast context but inherits the bf16
                             ops recorded in the forward, so the speed/memory win is kept.
            mixed_precision_dtype: "bfloat16" (default, no GradScaler needed) or "float16".
        """
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.colorfulness_weight = colorfulness_weight
        self.colorfulness_target = colorfulness_target
        self.use_lpips = use_lpips
        self.device = device
        self.input_channel = input_channel
        self.target_channel = target_channel
        self.mixed_precision = mixed_precision
        if mixed_precision_dtype not in _AUTOCAST_DTYPES:
            raise ValueError(f"mixed_precision_dtype must be one of {sorted(_AUTOCAST_DTYPES)}, "
                             f"got {mixed_precision_dtype!r}")
        self.mixed_precision_dtype = _AUTOCAST_DTYPES[mixed_precision_dtype]

        self.perceptual_loss_fn = None
        if use_lpips and perceptual_weight > 0:
            self.perceptual_loss_fn = lpips.LPIPS(net=lpips_net).to(device)
            self.perceptual_loss_fn.eval()
            for param in self.perceptual_loss_fn.parameters():
                param.requires_grad = False

    def __call__(
        self,
        models: Dict[str, nn.Module],
        batch: Tuple,
        metrics: object,
        gradientClip: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate generator loss and update metrics.

        Args:
            models: Dictionary containing the generator model (e.g., {'generator': generator_model})
            batch: Tuple of (modified_images, original_images)
            metrics: Metrics object with accumulate() method
            gradientClip: Gradient clipping value (None to disable)

        Returns:
            Dictionary with 'loss' key containing the total loss tensor
        """
        if 'generator' not in models:
            raise ValueError("models dictionary must contain 'generator' key")
        generator = models['generator']

        modifiedImages, originalImages = batch[0], batch[1]
        # nullcontext (not autocast(enabled=False)) so the fp32 path is untouched: merely entering
        # a disabled autocast region perturbs kernel selection by ~1 ULP.
        forwardContext = (torch.autocast(device_type=self.device.type,
                                         dtype=self.mixed_precision_dtype)
                          if self.mixed_precision else contextlib.nullcontext())
        with forwardContext:
            reconstructedImages = generator(modifiedImages)
        # Every loss term below runs in fp32 on purpose -- see mixed_precision in __init__.
        if reconstructedImages.dtype != torch.float32:
            reconstructedImages = reconstructedImages.float()

        reconstructionLoss = self.calculateReconstructionLoss(
            reconstructedImages, originalImages
        )

        perceptualLoss = torch.tensor(0.0, device=reconstructionLoss.device)
        if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
            perceptualLoss = self.calculatePerceptualLoss(
                reconstructedImages, originalImages, modifiedImages
            )

        colorfulnessLoss, colorfulnessRecon, colorfulnessOriginal = self.calculateColorfulnessLoss(
            reconstructedImages, originalImages, modifiedImages
        )

        totalLoss = (self.recon_weight * reconstructionLoss +
                     self.perceptual_weight * perceptualLoss +
                     self.colorfulness_weight * colorfulnessLoss)

        if torch.is_grad_enabled():
            totalLoss.backward()

            if gradientClip is not None:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=gradientClip)

        metrics.accumulate({
            'total_loss': totalLoss.item(),
            'reconstruction_loss': reconstructionLoss.item() * self.recon_weight,
            'perceptual_loss': perceptualLoss.item() * self.perceptual_weight,
            'colorfulness_loss': colorfulnessLoss.item() * self.colorfulness_weight,
            'colorfulness_recon': colorfulnessRecon.item(),
            'colorfulness_original': colorfulnessOriginal.item()
        })

        return {
            'loss': totalLoss,
            'reconstruction_loss': reconstructionLoss * self.recon_weight,
            'perceptual_loss': perceptualLoss * self.perceptual_weight,
            'colorfulness_loss': colorfulnessLoss * self.colorfulness_weight,
            'reconstructed_images': reconstructedImages
        }

    def calculateReconstructionLoss(self, reconstructed, original):
        """Calculate reconstruction loss based on channel types"""
        return self.reconstruction_loss_fn(reconstructed, original)

    def calculatePerceptualLoss(self, reconstructed, original, modified):
        """Calculate perceptual loss, converting to RGB if needed"""
        # Convert to RGB for perceptual loss calculation
        recon_rgb = self._convertToRgbForLoss(reconstructed, modified)
        original_rgb = self._convertToRgbForLoss(original, modified)

        # LPIPS expects images in range [-1, 1]
        recon_normalized = recon_rgb * 2.0 - 1.0
        original_normalized = original_rgb * 2.0 - 1.0

        return self.perceptual_loss_fn(recon_normalized, original_normalized).mean()

    def calculateColorfulnessLoss(self, reconstructed, original, modified):
        """Calculate colorfulness loss, converting to RGB if needed"""

        colorfulnessLoss = torch.tensor(0.0, device=reconstructed.device)
        colorfulnessRecon = torch.tensor(0.0, device=reconstructed.device)
        colorfulnessOriginal = torch.tensor(0.0, device=reconstructed.device)

        if self.colorfulness_weight > 0:
            recon_rgb = self._convertToRgbForLoss(reconstructed, modified)
            original_rgb = self._convertToRgbForLoss(original, modified)

            colorfulnessRecon = calculateColorfulnessLoss(recon_rgb)
            colorfulnessOriginal = calculateColorfulnessLoss(original_rgb)

            if self.colorfulness_target is not None:
                target = torch.tensor(self.colorfulness_target, device=reconstructed.device)
                colorfulnessLoss = torch.abs(colorfulnessRecon - target)
            else:
                colorfulnessLoss = torch.abs(colorfulnessOriginal - colorfulnessRecon)

        return colorfulnessLoss, colorfulnessRecon, colorfulnessOriginal

    def _convertToRgbForLoss(self, images, modified):
        """Convert images to RGB for perceptual/colorfulness losses"""
        if self.target_channel == "RGB":
            return images
        elif self.target_channel == "luminance":
            return images.repeat(1, 3, 1, 1)
        elif self.target_channel == "AB":
            # For AB channels, we need L channel to convert to RGB
            if self.input_channel == "luminance" and modified.shape[1] == 1:
                # Colorization case: use modified (L) + images (AB)
                return labToRgb(modified * 100.0, images)  # L: [0,1] -> [0,100]
            elif self.input_channel == "RGB" and modified.shape[1] == 3:
                # AB prediction case: use original image's L + images (AB)
                l_channel = self._rgb_to_luminance(modified) * 100.0
                return labToRgb(l_channel, images)
            else:
                # Fallback: use middle-gray L
                fake_l = torch.ones_like(images[:, 0:1]) * 50.0  # [0,100] range
                return labToRgb(fake_l, images)
        elif self.target_channel == "LAB":
            # Full LAB to RGB
            return labToRgbForVisualization(images)
        elif self.target_channel in ("LAB_A", "LAB_B"):
            # Single A or B channel in Kornia's native [-127,127] range
            ab_channel = images

            if self.input_channel == "luminance" and modified.shape[1] == 1:
                # Use modified (L) as luminance
                l_channel = modified * 100.0  # L: [0,1] -> [0,100]
            elif self.input_channel == "RGB" and modified.shape[1] == 3:
                l_channel = self._rgb_to_luminance(modified) * 100.0
            else:
                # Fallback: use middle-gray L
                l_channel = torch.ones_like(ab_channel) * 50.0  # [0,100] range

            zeros = torch.zeros_like(ab_channel)
            if self.target_channel == "LAB_A":
                ab_channels = torch.cat([ab_channel, zeros], dim=1)
            else:
                ab_channels = torch.cat([zeros, ab_channel], dim=1)

            return labToRgb(l_channel, ab_channels)
        elif self.target_channel == "R":
            return torch.cat([images, torch.zeros_like(images), torch.zeros_like(images)], dim=1)
        elif self.target_channel == "G":
            return torch.cat([torch.zeros_like(images), images, torch.zeros_like(images)], dim=1)
        elif self.target_channel == "B":
            return torch.cat([torch.zeros_like(images), torch.zeros_like(images), images], dim=1)
        else:
            return images.repeat(1, 3, 1, 1)

    def _rgb_to_luminance(self, rgb_images):
        """Convert RGB to luminance (L channel) using standard weights"""
        # RGB in [0,1] range
        luminance = 0.299 * rgb_images[:, 0:1] + 0.587 * rgb_images[:, 1:2] + 0.114 * rgb_images[:, 2:3]
        return luminance


class SegmentationDiceCELoss:
    """
    Dice + Cross Entropy Loss from monai
    """
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5
    ):
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.loss_fn = DiceCELoss(
            to_onehot_y=True,      # Class indices to one-hot
            softmax=True,          # Softmax for multi-class
            lambda_dice=dice_weight,
            lambda_ce=ce_weight
        )

    def __call__(
        self,
        models: Dict[str, nn.Module],
        batch: Tuple,
        metrics: Metrics,
        gradient_clip: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate Dice + CE loss
        """
        model = models['segformer']
        input_volumes, seg_labels = batch[0], batch[1]

        # [B, C, D, H, W]
        seg_logits = model(input_volumes)

        total_loss = self.loss_fn(seg_logits, seg_labels)

        if torch.is_grad_enabled():
            total_loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        metrics.accumulate({
            'total_loss': total_loss.item(),
        })

        return {
            'loss': total_loss,
        }
