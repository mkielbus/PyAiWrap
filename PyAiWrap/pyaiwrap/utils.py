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
