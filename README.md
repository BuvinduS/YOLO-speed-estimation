# Car Detection & Speed Estimation using Computer Vision

This project is a **computer vision prototype** for detecting car among other vehicles and estimating their speed from video footage using a single fixed camera.

It uses **YOLOv8** for object detection, **BoT-SORT** for multi-object tracking, and **OpenCV** for video processing. The system is designed as a proof-of-concept and is currently tested on publicly available footage.

---

## Features

- Vehicle detection using pretrained YOLOv8 models (n / s / m)
- Multi-object tracking with BoT-SORT
- Speed estimation using a **two-line timing method**
- Vehicle counting using a virtual line
- CUDA-accelerated inference for near real-time performance
- Video-based visualization with bounding boxes, IDs, and speed estimates

---

## Speed Estimation Method

Speed is estimated using a **two-line approach**:

- Timing starts when a vehicle crosses the **red line**
- Timing ends when the vehicle crosses the **blue line**
- Vehicles are counted when they cross the **green line**

The distance between the two speed-measurement lines is currently **assumed to be 15 meters**.  
In a real-world deployment, this distance would be accurately measured on-site.

> ⚠️ Because the system relies on a single camera view, factors such as camera placement, road curvature, and perspective distortion mean that the estimated speeds are **approximations** and remain video-dependent.

---

## 🛠️ Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- BoT-SORT (via Ultralytics tracking)
- NumPy

---

## ✅ Prerequisites

- Python **3.9+** recommended
- NVIDIA GPU with CUDA support (recommended)
- CUDA-compatible PyTorch installation for best performance

> The project will run on CPU-only systems, but inference will be significantly slower.

---

## ⚙️ Installation

### Option 1: Quick Setup (Recommended)

1) Clone the repository:
```bash
git clone https://github.com/BuvinduS/YOLO-speed-estimation
cd YOLO-speed-estimation
```
2) Create a virtual environment (only needed to run once):
```bash
python -m venv .venv
```
Activate the virtual environment (virtual environment must be activated before proceeding):
```bash
.venv\Scripts\activate.bat
```

3) Install the core dependencies:

> ⚠️ Ensure a virtual environment is activated before installing dependencies, global dependency installation is NOT recommended.

```bash
pip install cvzone ultralytics
```
Then install PyTorch with CUDA support by following the official PyTorch instructions:

https://pytorch.org/get-started/locally/

Make sure to select:
- Your OS
- pip
- CUDA (do not select CPU-only unless you do not have a dedicated GPU)

The command given will be similar to the following:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
> Note: The exact CUDA version (`cuXXX`) depends on your GPU and system configuration.

### Option 2: Using ```requirements.txt```

1) Clone the repository:
```bash
git clone https://github.com/BuvinduS/YOLO-speed-estimation
cd YOLO-speed-estimation
```
2) Create a virtual environment (only needed to run once):
```bash
python -m venv .venv
```
Activate the virtual environment (virtual environment must be activated before proceeding):
```bash
.venv\Scripts\activate.bat
```

3) Install the listed dependencies:

> ⚠️ Ensure a virtual environment is activated before installing dependencies, global dependency installation is NOT recommended.

```bash
pip install -r requirements.txt
```
Then reinstall PyTorch with CUDA support (pip installs CPU-only versions by default):
```bash
pip uninstall torch torchvision torchaudio
```

Then install PyTorch with CUDA support by following the official PyTorch instructions:

https://pytorch.org/get-started/locally/

Make sure to select:
- Your OS
- pip
- CUDA (do not select CPU-only unless you do not have a dedicated GPU)

The command given will be similar to the following:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
> Note: The exact CUDA version (`cuXXX`) depends on your GPU and system configuration.

## Running the Project

Once all dependencies are installed cd to the root and run:
```bash
python YOLO_main.py
```

## ⚠️ Hardware & CUDA Notes
**CUDA** is **strongly** recommended for usable performance.
- AMD GPUs are not officially supported by PyTorch CUDA builds.

- If you do not have an NVIDIA GPU, consider:

- Running in CPU mode (slow)

- Using Google Colab or another cloud GPU environment

## Limitations
- Speed estimates are approximate and video-dependent

- Perspective distortion is not compensated (no homography applied)

- Pretrained models may struggle in dense traffic or distant camera views

- Real-world distances are assumed, not measured

## Potential Improvements
- Fine-tuning YOLO models on custom traffic datasets

- Perspective correction using homography

- Improved calibration for real-world distance estimation

- Refactoring using higher-level computer vision libraries (e.g., `supervision`)
  
- Potential integration with IoT or edge deployment pipelines
