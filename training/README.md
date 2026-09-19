# Mudra — Training

This guide walks you through setting up the offline Python training environment for Mudra.
The training pipeline processes the raw dataset, extracts hand landmarks, trains the main PyTorch MLP, and exports it to ONNX.

## Python Version
Python **3.11** is recommended for compatibility with all dependencies (especially `mediapipe` and `onnxruntime`). 
Python 3.10 and 3.12 are also acceptable.

## Setting up the Virtual Environment

### Windows
1. Open PowerShell or Command Prompt in the `training/` directory.
2. Create the virtual environment: `python -m venv .venv`
3. Activate the virtual environment: `.\.venv\Scripts\activate`

### macOS / Linux
1. Open your terminal in the `training/` directory.
2. Create the virtual environment: `python3 -m venv .venv`
3. Activate the virtual environment: `source .venv/bin/activate`

## Installing Requirements

Locally, you only need the CPU build of PyTorch, as the main landmark MLP trains in minutes on a standard CPU. The MobileNetV2 comparison model is intended to be trained on Google Colab's free GPU.

**Install the CPU version of PyTorch and all dependencies:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

*(Note: If you have an NVIDIA GPU and wish to train locally with CUDA, simply install PyTorch without the `--index-url` flag using the default index, according to the official PyTorch instructions.)*

## Verifying the Installation

After installing, verify that everything is set up correctly by running this one-line command:
```bash
python -c "import torch, torchvision, onnxruntime, mediapipe, cv2; print(f'PyTorch: {torch.__version__} | TorchVision: {torchvision.__version__} | ONNXRuntime: {onnxruntime.__version__} | MediaPipe: {mediapipe.__version__} | OpenCV: {cv2.__version__}')"
```

## Workflow
*(To be filled in Section 12)*
