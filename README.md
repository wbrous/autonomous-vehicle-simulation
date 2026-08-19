# Camera ACC Demo

Real-time adaptive cruise control from a single forward-facing camera — YOLO detection, YOLOPv2 lane tracking, and an advisory brake/speed HUD, served headless in the browser.

![Camera ACC Demo](docs/images/hero.png)

## What it does

A live camera feed is turned into a driving-assistance overlay. Every vehicle, pedestrian, traffic sign, and traffic light is detected and tracked; distances are estimated with a pinhole camera model; the ego lane is traced from YOLOPv2 lane masks; and a decision engine produces an advisory action — `HEAVY_BRAKE`, `LIGHT_BRAKE`, or `NO_SUGGESTION` — rendered on top of the stream.

## Features

- **Unified YOLO detection** — one model for 8 classes: car, truck, bus, motorcycle, bicycle, person, traffic sign, traffic light.
- **Pinhole distance estimation** — per-class reference heights turn bounding-box size into metres.
- **Dynamic ego-lane tracking** — YOLOPv2 lane masks are reduced to a drivable lane, locked only where both edges are actually visible.
- **Hazard reasoning** — closing-rate braking, crosswalk and jaywalking pedestrians, and crossing traffic at intersections.
- **Headless web UI** — the annotated stream is served over HTTP; no display window required.

## Screenshots

![Detection with distance labels](docs/images/detection.png)

![Ego-lane overlay](docs/images/lanes.png)

![ACC decision HUD](docs/images/hud.png)

## Demo

https://github.com/user-attachments/assets/29aff393-a864-4848-8c4e-a2a53b9a0ea5

*Player not rendering? Open [docs/videos/demo.mp4](docs/videos/demo.mp4).*

## How it works

```mermaid
flowchart LR
    Cam[Camera] --> Det[YOLO detection]
    Cam --> Lane[YOLOPv2 lanes]
    Det --> Track[BoxTracker]
    Lane --> Ego[Ego-lane mask]
    Track --> Decide[ACC decision engine]
    Ego --> Decide
    Decide --> Web[Annotated web stream]
```

1. **Detect** — the unified model localises the 8 classes every frame.
2. **Track** — `BoxTracker` associates detections across frames and estimates lateral and vertical motion.
3. **Measure** — distance comes from bounding-box height and the pinhole focal length.
4. **Lane** — the YOLOPv2 lane mask is scanned bottom-up to extract a per-row ego lane.
5. **Decide** — closing rate, crosswalk and jaywalker hazards, and crossing traffic set the advisory action.

## Quick start

```bash
pip install -r requirements.txt
# plus the PyTorch backend for your hardware (see below)
python camera_demo_acc_web.py
```

Open http://localhost:5000.

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Camera device index |
| `--width` | `1280` | Capture width |
| `--height` | `720` | Capture height |
| `--imgsz` | `640` | Detection inference size |

All tunables — weights path, focal length, ACC zones, hazard thresholds — live in the `CONFIG` section at the top of `camera_demo_acc_web.py`.

## Model weights

The two model files are committed directly in `weights/` (no longer gitignored):

| File | Purpose |
|------|---------|
| `weights/yolo12m_vistas_best.pt` | YOLO detection (8 classes) |
| `weights/yolopv2.pt` | YOLOPv2 lane detection |

## Hardware backends

`requirements.txt` pins `ultralytics` and `flask`. Install the PyTorch backend for your hardware:

| Backend | Command |
|---------|---------|
| NVIDIA CUDA | `pip install torch torchvision` |
| AMD ROCm (RX 7000) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2` |
| CPU only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

## Project layout

```
.
├── camera_demo_acc_web.py   # active demo application
├── main.py                  # training pipeline
├── train_*.py               # training variants
├── build_dataset.py         # Roboflow dataset builder
├── *_to_yolo.py             # dataset converters
├── weights/                 # model weights (committed)
├── docs/                    # docs assets
│   ├── images/              # screenshots
│   └── videos/              # demo video
├── requirements.txt
├── .env.example
└── archive/                 # superseded demos, datasets, eval scripts
```

## Challenges

Problems I ran into building this, and how I fixed them:

- **ROCm crash on the wrong GPU.** Training kept crashing with `Module not initialized` from `hip_global.cpp`. It turned out my RX 7900 XT was sharing device visibility with the CPU's integrated Raphael graphics, and PyTorch's ROCm dataloaders don't survive multiprocessing either. I fixed both by setting `HIP_VISIBLE_DEVICES=0` before `torch` gets imported and forcing `workers=0` on ROCm.
- **85/15 class imbalance.** The first model I trained was basically useless in practice since I'd capped the dataset at 2,000 images without checking the split of person <-> vehicle, so it came out 85% pedestrian boxes to 15% vehicle. Additionally, it was trained for only 50 epochs with no augmentation tuning, which left the model in a bad state after I'd finished the training. To fix it, I rebuilt `build_dataset.py` to pull from 9 Roboflow sources plus a local dataset with per-source class remapping, dropped the image cap entirely, and retrained at 100 epochs with mosaic/mixup/copy-paste/HSV augmentation turned on.
- **Adjacent-lane false positives.** There was also a lane-detection bug since `track_in_lane` only checked a box's bottom-center against the lane trapezoid, so a car in the next lane over would register as in-lane once it was far enough away that the trapezoid had narrowed. I fixed it by adding a second check on the box's top-center, gated to the horizon line, so close-range detection still works.
- **Screenshot crash.** Screenshots crashed with `NotImplementedError` because `pygame.image.save()` can't handle the `DOUBLEBUF | RESIZABLE` surface the demo runs on, so I switched to writing the OpenCV frame directly with `cv2.imwrite` instead.
- **Leaked API key and repo bloat.** At some point I realized I'd hardcoded a Roboflow API key in two files, and the repo had grown to ~30GB of old scripts, cloned repos, and stale checkpoints, so I moved the key into `.env`, archived what wasn't in use anymore, and deleted the untracked weight files.

## Future additions

Things that are still open:

- **Real ego-speed input.** The HUD's `HEAVY_BRAKE`/`LIGHT_BRAKE` actions are advisory only, there's no speed sensor or actuation behind them. Wiring in an OBD-II reader (or even GPS-derived speed) would let the decision engine reason about actual closing speed instead of just distance trend.
- **Adjacent-lane awareness.** The decision engine only looks at the lead vehicle in the ego lane. A car drifting into the lane from the side, or a merge, isn't part of the reasoning yet.
- **Raspberry Pi deployment.** The pinhole distance math is cheap enough to run on a Pi, but getting there means exporting the detection model to NCNN, swapping `cv2.VideoCapture` for `picamera2`, and confirming framerate holds at 320px input.
- **Better distance estimation.** Pinhole distance from bounding-box height is only accurate to about ±15-20%, and it degrades if the camera mount isn't level. Ground-plane geometry or a stereo pair would tighten that up, at the cost of needing a fixed, calibrated mount.

## Training & datasets

Training and dataset-building scripts are kept at the root (`main.py`, `train_*.py`, `build_*.py`, `*_to_yolo.py`). They require extra dependencies beyond the demo runtime: `roboflow`, `matplotlib`, `onnxruntime`. Set `ROBOFLOW_API_KEY` in `.env` (copy `.env.example`) for the Roboflow-based builders.
