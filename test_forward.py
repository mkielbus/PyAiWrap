import os
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from pyaiwrap.utils import prepareDevice
from pyaiwrap.neural_network import ColorizationTransformerNet


def main():
    # 1. Urządzenie
    device = prepareDevice(use_cuda=True)

    # 2. Ścieżki – teraz bierzemy dane z data/test_images
    input_dir = Path("data/test_images")
    output_dir = Path("data/test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Transformacja: zmiana rozmiaru + tensor
    image_size = 256
    to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),   # [C,H,W] w [0,1]
    ])
    to_pil = transforms.ToPILImage()

    model = ColorizationTransformerNet(
        embed_dim=256,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        num_layers=4,
        num_color_tokens=64,     # 8x8
        num_image_patches=64,    # 8x8
        image_size=image_size,
        use_decoder_masking=False,
        only_use_encoder=False,
        output_channels=3,
    ).to(device)

    # 🔹 Wczytujemy najlepsze wagi z treningu
    weights_path = "weights/best_performance_color_transformer_generator_hyperparams_first_run.pth"
    if os.path.exists(weights_path):
        print(f"Ładuję wagi z: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"UWAGA: nie znaleziono {weights_path}, używam losowej inicjalizacji.")

    model.eval()

    # 5. Przelot po wszystkich obrazkach
    with torch.no_grad():
        for img_path in input_dir.iterdir():
            if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                continue

            print(f"Przetwarzam: {img_path.name}")
            img = Image.open(img_path).convert("RGB")
            inp = to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W]

            # Forward przez model
            out = model(inp)  # [1,3,H,W]

            # Przycinamy do [0,1] i zapisujemy
            out = out.clamp(0.0, 1.0).squeeze(0).cpu()   # [3,H,W]

            out_img = to_pil(out)
            out_path = output_dir / f"{img_path.stem}_out.png"
            out_img.save(out_path)
            print(f"Zapisano: {out_path}")


if __name__ == "__main__":
    main()
