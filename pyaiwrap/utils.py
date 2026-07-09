import torch


def prepareDevice(use_cuda: bool = True, device_id: int = 0) -> torch.device:
    """
    Prepare and return the appropriate device (CUDA, MPS, or CPU).

    Args:
        use_cuda (bool): Whether to use CUDA if available (default: True)
        device_id (int): CUDA device ID to use (default: 0)

    Returns:
        torch.device: The device to use for training/inference
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def sinusoidalPositionalEncoding2D(height: int, width: int, d_model: int, device: torch.device):
    """
    Standard 2D sinusoidal positional encoding (as in MAE/ViT): half the channels
    encode the y coordinate and half the x coordinate, each half using sin/cos pairs
    with the geometric frequency ladder from "Attention Is All You Need"
    (wavelengths from 2*pi to 10000*2*pi over integer grid positions).
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by 4 for 2D sinusoidal encoding")

    num_frequencies = d_model // 4
    omega = torch.arange(num_frequencies, device=device).float() / num_frequencies
    omega = 1.0 / (10000 ** omega)  # [d_model/4]

    y_coords = torch.arange(height, device=device).float()
    x_coords = torch.arange(width, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    y_angles = y_grid.reshape(-1, 1) * omega  # [height*width, d_model/4]
    x_angles = x_grid.reshape(-1, 1) * omega  # [height*width, d_model/4]

    encoding = torch.cat([
        torch.sin(y_angles), torch.cos(y_angles),
        torch.sin(x_angles), torch.cos(x_angles)
    ], dim=1)  # [height*width, d_model]

    return encoding.unsqueeze(0)  # [1, height*width, d_model]
