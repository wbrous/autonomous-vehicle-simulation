# Camera ACC Demo

Real-time adaptive cruise control (ACC) demo for a forward-facing camera. Detects vehicles, pedestrians, and traffic signals with YOLO, estimates distance via a pinhole camera model, tracks the ego lane with YOLOPv2, and renders an advisory brake/speed HUD. Headless: serves the annotated video stream over HTTP rather than opening an OpenCV window.

## Run the demo

```bash
pip install -r requirements.txt
# plus the PyTorch backend for your hardware (see below)
python camera_demo_acc_web.py
```

Open http://localhost:5000 to view the live annotated stream.

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Camera device index |
| `--width` | `1280` | Capture width |
| `--height` | `720` | Capture height |
| `--imgsz` | `640` | Detection inference size |

All tunables (weights path, focal length, ACC zones, hazard thresholds) live in the `CONFIG` section near the top of `camera_demo_acc_web.py`.

## Runtime dependencies

`requirements.txt` pins `ultralytics` and `flask`. Install the PyTorch backend for your hardware separately:

| Backend | Command |
|---------|---------|
| NVIDIA CUDA | `pip install torch torchvision` |
| AMD ROCm (RX 7000 series) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2` |
| CPU only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

Required model files (gitignored): `weights/yolo12m_vistas_best.pt` (detection) and `weights/yolopv2.pt` (lane detection).

## Training and dataset building

The training and dataset-building scripts are kept at the root:

- `main.py` — YOLO training pipeline (Roboflow dataset)
- `train_signs.py`, `train_lisa.py`, `train_actual_lisa_signs.py` — sign/LISA training variants
- `build_dataset.py`, `build_bdd10k_dataset.py`, `vistas_to_yolo.py`, `mtsd_to_yolo.py`, `create_dataset.py`, `extract_signs.py` — dataset builders/converters

These need extra dependencies beyond the demo runtime: `roboflow`, `matplotlib`, `onnxruntime`.

## Environment

Copy `.env.example` to `.env` and set `ROBOFLOW_API_KEY`. It is read by `build_dataset.py` and `main.py`; the key is kept out of source control.

## Layout

```
.
├── camera_demo_acc_web.py   # active demo application
├── main.py                  # training pipeline
├── build_dataset.py         # Roboflow dataset builder
├── *_to_yolo.py             # dataset converters
├── train_*.py               # training variants
├── weights/                 # model weights: yolo12m_vistas_best.pt, yolopv2.pt (gitignored)
├── requirements.txt
├── .env.example
└── archive/                 # superseded demos, datasets, and eval scripts
```
