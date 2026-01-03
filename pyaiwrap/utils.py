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


def distancePositionalEncoding(height: int, width: int, d_model: int, device: torch.device):
    """2D positional encoding based on distance from center"""
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    distance = torch.sqrt(y_grid**2 + x_grid**2)  # [height, width]
    angle = torch.atan2(y_grid, x_grid)  # [height, width]

    distance_flat = distance.reshape(-1, 1)  # [height*width, 1]
    angle_flat = angle.reshape(-1, 1)        # [height*width, 1]

    dimensions = torch.arange(d_model, device=device).float().unsqueeze(0)  # [1, d_model]

    encoding = torch.zeros(height * width, d_model, device=device)

    encoding[:, 0::2] = torch.sin(distance_flat * dimensions[:, 0::2] * torch.pi)

    encoding[:, 1::2] = torch.cos(angle_flat * dimensions[:, 1::2] * torch.pi)

    return encoding.unsqueeze(0)  # [1, height*width, d_model]
