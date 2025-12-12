import os
from typing import List

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pyaiwrap.utils import prepareDevice
from pyaiwrap.neural_network import ColorizationTransformerNet
from run_fusion_training import SimpleFusionNet  # użyjemy tej samej klasy


def load_single_ctn(weights_path: str, device: torch.device) -> nn.Module:
    image_size = 128
    model = ColorizationTransformerNet(
        embed_dim=128,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.1,
        num_layers=2,
        num_color_tokens=1024,
        num_image_patches=1024,
        image_size=image_size,
        use_decoder_masking=False,
        only_use_encoder=False,
        output_channels=1,
    ).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_fusion(weights_path: str, device: torch.device) -> nn.Module:
    model = SimpleFusionNet(in_channels=3, hidden_channels=32).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """
    t: [3,H,W], w zakresie [0,1]
    """
    t = t.clamp(0, 1)
    t = (t * 255).byte().cpu()
    return transforms.ToPILImage()(t)


def main():
    device = prepareDevice(use_cuda=True)

    weights_dir = "weights"
    ctn_r_path = os.path.join(weights_dir, "ctn_r_best.pth")
    ctn_g_path = os.path.join(weights_dir, "ctn_g_best.pth")
    ctn_b_path = os.path.join(weights_dir, "ctn_b_best.pth")
    fusion_path = os.path.join(weights_dir, "fusion_best.pth")

    assert os.path.exists(ctn_r_path), "Brak ctn_r_best.pth"
    assert os.path.exists(ctn_g_path), "Brak ctn_g_best.pth"
    assert os.path.exists(ctn_b_path), "Brak ctn_b_best.pth"

    ctn_r = load_single_ctn(ctn_r_path, device)
    ctn_g = load_single_ctn(ctn_g_path, device)
    ctn_b = load_single_ctn(ctn_b_path, device)

    fusion_net = None
    if os.path.exists(fusion_path):
        print(f"Załadowano FusionNet z wagami: {fusion_path}")
        fusion_net = load_fusion(fusion_path, device)
    else:
        print("Uwaga: brak fusion_best.pth – pokazuję tylko wynik 3×CTN bez fuzji.")

    # ------------------ OBRAZKI TESTOWE ------------------
    test_dir = "data/test_images"
    image_files: List[str] = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()
    if not image_files:
        raise RuntimeError(f"Brak obrazów w {test_dir}")

    os.makedirs("outputs_ctn", exist_ok=True)

    transform_input = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),      # [C,H,W], [0,1]
    ])

    for idx, filename in enumerate(image_files):
        path = os.path.join(test_dir, filename)
        img = Image.open(path).convert("RGB")
        img_tensor = transform_input(img).unsqueeze(0).to(device)  # [1,3,H,W]

        # szarość
        gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]

        with torch.no_grad():
            r = ctn_r(gray)
            g = ctn_g(gray)
            b = ctn_b(gray)

            rgb_from_ctn = torch.cat([r, g, b], dim=1)  # [1,3,H,W]
            rgb_from_ctn_clamped = rgb_from_ctn.clamp(0, 1)

            if fusion_net is not None:
                rgb_fused = fusion_net(rgb_from_ctn_clamped)
                rgb_fused = rgb_fused.clamp(0, 1)
            else:
                rgb_fused = rgb_from_ctn_clamped

        # konwersja do obrazów
        orig_img = tensor_to_image(img_tensor[0])
        gray_img = tensor_to_image(gray.repeat(1, 3, 1, 1)[0])  # szarość w 3 kanałach
        ctn_img = tensor_to_image(rgb_from_ctn_clamped[0])
        fused_img = tensor_to_image(rgb_fused[0])

        # rysunek 2x2
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title("Oryginał")
        axes[0].axis("off")

        axes[1].imshow(gray_img)
        axes[1].set_title("Szarość")
        axes[1].axis("off")

        axes[2].imshow(ctn_img)
        axes[2].set_title("3×CTN (stack)")
        axes[2].axis("off")

        axes[3].imshow(fused_img)
        axes[3].set_title("FusionNet")
        axes[3].axis("off")

        plt.tight_layout()
        out_path = os.path.join("outputs_ctn", f"ctn_fusion_example_{idx:02d}.png")
        plt.savefig(out_path)
        plt.close(fig)

        print(f"Zapisano: {out_path}")


if __name__ == "__main__":
    main()
