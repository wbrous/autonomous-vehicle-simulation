# YOLOv8 Car Training

YOLOv8 Nano training pipeline using a Roboflow dataset.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train
python main.py
```

## Hardware Backends

The script auto-detects the best available device.

| Vendor | Install Command | Notes |
|--------|----------------|-------|
| **NVIDIA CUDA** | `pip install torch torchvision` | Standard PyTorch build |
| **AMD ROCm** | See below | RX 7000 series, MI series |
| **Apple MPS** | `pip install torch torchvision` | M1/M2/M3 Macs |
| **CPU** | `pip install torch torchvision` | Slow but works |

### AMD ROCm Setup (e.g. RX 7900 XT)

```bash
# Install ROCm PyTorch (Linux only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Verify
gpu-detect
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

> Replace `rocm6.2` with the latest version from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Configuration

All tunables are at the top of `main.py`:

```python
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
LEARNING_RATE = 0.01
PATIENCE = 50
WORKERS = os.cpu_count() or 1  # all cores
```

## Project Structure

```
.
├── main.py              # training script
├── requirements.txt     # dependencies
└── yolov8-car-training/ # outputs (created at runtime)
    └── nano-run-1/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        └── predictions/
```
