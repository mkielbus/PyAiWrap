import torch
import math


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


def getDeviceInfo(device: torch.device) -> dict:
    """
    Get detailed information about the device.

    Args:
        device (torch.device): The device to query

    Returns:
        dict: Dictionary containing device information
    """
    info = {
        "type": device.type,
        "index": device.index if device.type == "cuda" else None
    }

    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "cuda_version": torch.version.cuda
        })

    return info


def clearCache(device: torch.device) -> None:
    """
    Clear CUDA cache if using GPU.

    Args:
        device (torch.device): The device to clear cache for
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"Cleared CUDA cache for device {device}")
    elif device.type == "mps":
        torch.mps.empty_cache()
        print("Cleared MPS cache")


def setDevice(device_id: int = 0, force_cpu: bool = False) -> torch.device:
    """
    Set and return a specific device with more control options.

    Args:
        device_id (int): CUDA device ID (default: 0)
        force_cpu (bool): Force CPU usage even if GPU is available (default: False)

    Returns:
        torch.device: The configured device
    """
    if force_cpu:
        device = torch.device("cpu")
        print("Forced CPU usage")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        print(f"Set device to: {torch.cuda.get_device_name(device_id)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Set device to: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Set device to: CPU (no GPU available)")

    return device


def getAvailableDevices() -> list:
    """
    Get a list of all available devices.

    Returns:
        list: List of available torch.device objects
    """
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))

    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    return devices


def printDeviceInfo(device: torch.device) -> None:
    """
    Print detailed information about a device.

    Args:
        device (torch.device): Device to print information about
    """
    print(f"\n{'='*50}")
    print("Device Information")
    print(f"{'='*50}")
    print(f"Device type: {device.type}")

    if device.type == "cuda":
        print(f"Device index: {device.index}")
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        props = torch.cuda.get_device_properties(device)
        print(f"Compute capability: {props.major}.{props.minor}")
        print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    elif device.type == "mps":
        print("Apple Silicon GPU (Metal Performance Shaders)")
    else:
        print("CPU - No GPU acceleration")

    print(f"PyTorch version: {torch.__version__}")
    print(f"{'='*50}\n")


def enableTF32(enable: bool = True) -> None:
    """
    Enable or disable TF32 for CUDA operations (Ampere GPUs and newer).
    TF32 can speed up training with minimal accuracy loss.

    Args:
        enable (bool): Whether to enable TF32 (default: True)
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        print(f"TF32 {'enabled' if enable else 'disabled'}")
    else:
        print("TF32 is only available on CUDA devices")


def setBenchmark(enable: bool = True) -> None:
    """
    Enable or disable cuDNN benchmark mode for faster training.
    Use when input sizes are constant.

    Args:
        enable (bool): Whether to enable benchmark mode (default: True)
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable
        print(f"cuDNN benchmark mode {'enabled' if enable else 'disabled'}")
    else:
        print("cuDNN benchmark is only available on CUDA devices")


def setDeterministic(enable: bool = True) -> None:
    """
    Enable or disable deterministic mode for reproducibility.
    Note: This may reduce performance.

    Args:
        enable (bool): Whether to enable deterministic mode (default: True)
    """
    torch.use_deterministic_algorithms(enable)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = enable
        torch.backends.cudnn.benchmark = not enable
    print(f"Deterministic mode {'enabled' if enable else 'disabled'}")


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


def sinusoidal_position_encoding_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Klasyczne sinusoidalne kodowanie pozycyjne 1D.
    positions: [N] – indeksy pozycji (0,1,2,...)
    dim: wymiar wektora PE (musi być parzysty)
    zwraca: [N, dim]
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_position_encoding_1d: dim musi być parzyste, a jest {dim}")

    device = positions.device
    half_dim = dim // 2

    div_term = torch.exp(
        torch.arange(0, half_dim, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / half_dim)
    )  

    
    angles = positions.float().unsqueeze(1) * div_term.unsqueeze(0)

    pe = torch.zeros(positions.size(0), dim, device=device)
    pe[:, 0::2] = torch.sin(angles)   
    pe[:, 1::2] = torch.cos(angles)   
    return pe


def sinusoidal_position_encoding_2d(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    2D sinusoidalne PE (jak w ViT):
    - połowa wymiaru koduje pozycję w pionie (y)
    - połowa wymiaru koduje pozycję w poziomie (x)

    Zwraca tensor [1, height*width, dim] – dokładnie tak jak distancePositionalEncoding.
    """
    if dim % 4 != 0:
        raise ValueError(f"sinusoidal_position_encoding_2d: dim musi być podzielne przez 4, a jest {dim}")

  
    ys = torch.arange(height, device=device)
    xs = torch.arange(width, device=device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")  

    y_flat = y_grid.reshape(-1)
    x_flat = x_grid.reshape(-1)

    dim_h = dim // 2
    dim_w = dim // 2

    pe_y = sinusoidal_position_encoding_1d(y_flat, dim_h)  
    pe_x = sinusoidal_position_encoding_1d(x_flat, dim_w)  

    pe = torch.cat([pe_y, pe_x], dim=1)  
    return pe.unsqueeze(0)  


def ddcolor_position_encoding_2d(
    height: int,
    width: int,
    d_model: int,
    device: torch.device,
    temperature: float = 10000.0,
    normalize: bool = True,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """
    2D sine-cosine positional encoding w stylu DETR / Mask2Former,
    czyli taki typ kodowania, jaki używa też DDColor.

    Zwraca tensor [1, height*width, d_model].
    """
    
    if d_model % 4 != 0:
        raise ValueError(
            f"ddcolor_position_encoding_2d: d_model musi być podzielne przez 4, a jest {d_model}"
        )

    num_pos_feats = d_model // 2 

    y_embed = torch.arange(height, device=device).unsqueeze(1).repeat(1, width)
    x_embed = torch.arange(width, device=device).unsqueeze(0).repeat(height, 1)

    y_embed = y_embed.float()
    x_embed = x_embed.float()

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (height - 1 + eps) * scale
        x_embed = x_embed / (width - 1 + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats) 

    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t

    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=-1
    ).flatten(-2)

    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
        dim=-1
    ).flatten(-2) 

    pos = torch.cat([pos_y, pos_x], dim=-1)

    return pos.view(1, height * width, d_model)
