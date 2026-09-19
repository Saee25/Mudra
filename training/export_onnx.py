import os
import json
import shutil
import torch
import torch.onnx
import onnx
from datetime import datetime

from model_defs import LandmarkMLP, LandmarkExportWrapper

# Ensure we are in the training directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "..", "model")
FRONTEND_PUBLIC_MODEL = os.path.join(SCRIPT_DIR, "..", "frontend", "public", "model")

def main():
    print("Starting ONNX export...")
    
    # Check inputs
    ckpt_path = os.path.join(MODELS_DIR, "landmark_mlp.pt")
    class_names_path = os.path.join(OUTPUTS_DIR, "class_names.json")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
    if not os.path.exists(class_names_path):
        print(f"Error: Class names not found at {class_names_path}")
        return

    # Load class names
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    
    # Load model
    model = LandmarkMLP(num_classes=num_classes)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Wrap for export
    wrapped_model = LandmarkExportWrapper(model)
    
    # MUST call eval() before export!
    # Explanation: Eval mode is critical because LandmarkMLP contains BatchNorm1d and Dropout layers.
    # In training mode, BatchNorm uses batch statistics and updates running averages, and Dropout 
    # zeroes out random activations. In eval mode, BatchNorm uses its learned global stats and Dropout
    # is disabled, which is exactly what we want for deterministic inference in the browser.
    wrapped_model.eval()
    
    # Clear export directory
    os.makedirs(EXPORT_DIR, exist_ok=True)
    for f in os.listdir(EXPORT_DIR):
        if f != ".keep":
            os.remove(os.path.join(EXPORT_DIR, f))
    
    onnx_path = os.path.join(EXPORT_DIR, "mudra.onnx")
    
    # Create dummy inputs
    # Shape: [1, 2, 21, 3] for landmarks, [1, 2] for hand_mask
    dummy_landmarks = torch.randn(1, 2, 21, 3, dtype=torch.float32)
    dummy_mask = torch.ones(1, 2, dtype=torch.float32)
    
    # Export to ONNX
    # We choose the classic TorchScript-based exporter (torch.onnx.export) rather than the newer Dynamo exporter.
    # Why? Mudra's model is a tiny MLP with standard mathematical tensor operations (in LandmarkNormalization) 
    # and basic layers (Linear, BatchNorm, ReLU). The classic exporter handles this flawlessly, produces a clean,
    # highly optimized graph with opset 17, and has guaranteed compatibility with ONNX Runtime Web WASM backend.
    # Dynamo is powerful for complex dynamic control flow but can sometimes over-complicate simple math graphs.
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        wrapped_model,
        (dummy_landmarks, dummy_mask),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["landmarks", "hand_mask"],
        output_names=["probabilities"],
        dynamic_axes=None # Fixed shapes: B=1 for client-side web inference
    )
    
    # Check the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # File size and quantization explanation
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"ONNX export successful. Model checked. Size: {size_kb:.2f} KB.")
    print("No quantization (e.g. INT8) was applied. "
          "The model is already tiny (well under 1 MB). INT8 quantization would save a negligible amount "
          "of absolute bandwidth but carries a risk of degrading accuracy. For this scale, FP32 is best.")

    # Generate model_meta.json
    meta = {
        "model_name": "Mudra",
        "architecture": "LandmarkMLP",
        "pipeline_version": "landmark-v1",
        "inputs": [
            {"name": "landmarks", "shape": [1, 2, 21, 3], "dtype": "float32"},
            {"name": "hand_mask", "shape": [1, 2], "dtype": "float32"}
        ],
        "coordinate_convention": "x*width, y*height, z*width in pixels",
        "slot_rule": "sorted by wrist x ascending; single hand in slot 0; absent = zeros, mask 0",
        "output": {"name": "probabilities", "shape": [1, num_classes], "dtype": "float32"},
        "class_count": num_classes,
        "hand_detection": {
            "num_hands": 2,
            "min_detection_confidence": 0.3,
            "min_presence_confidence": 0.3
        },
        "crop_parameters": {
            "padding": 0.2,
            "shape": "square",
            "max_size": 224
        },
        "mirror_augment": True,
        "validation_accuracy": None, # Will be filled if we have it, or later
        "export_date": datetime.now().isoformat()
    }
    
    meta_path = os.path.join(EXPORT_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    # Copy class_names.json
    shutil.copy2(class_names_path, os.path.join(EXPORT_DIR, "class_names.json"))
    print("Wrote model_meta.json and class_names.json.")

if __name__ == "__main__":
    main()
