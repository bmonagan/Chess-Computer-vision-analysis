import torch
from torch import cuda,version

if torch.cuda.is_available():
    print("SUCCESS! PyTorch can use your CUDA-enabled GPU.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    current_device_id = torch.cuda.current_device()
    print(f"Current GPU ID: {current_device_id}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(current_device_id)}") # Should show "NVIDIA GeForce GTX 980"
    print(f"PyTorch CUDA version: {torch.version.cuda}") # type: ignore
    # For your GTX 980, the compute capability is 5.2
    major, minor = torch.cuda.get_device_capability(current_device_id)
    print(f"GPU Compute Capability: {major}.{minor}")

    # You can also assign a device object for later use
    device = torch.device("cuda")
    print(f"Running on device: {device}")

    # Simple test: create a tensor and move it to GPU
    try:
        x = torch.randn(3, 3).to(device)
        print("Successfully moved a tensor to the GPU:", x)
        print(f"Tensor is on CUDA: {x.is_cuda}")
    except Exception as e:
        print(f"Error during tensor test on GPU: {e}")

else:
    print("FAILURE. PyTorch cannot see or use your CUDA-enabled GPU.")
    print("Possible reasons:")
    print("- PyTorch was installed without CUDA support (CPU-only version).")
    print("- Incompatibility between PyTorch's CUDA version and your NVIDIA driver.")
    print("- Other environment or installation issues.")