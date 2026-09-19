import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "model")
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")

FRONTEND_PUBLIC_DIR = os.path.join(SCRIPT_DIR, "..", "frontend", "public")
FRONTEND_MODEL_DIR = os.path.join(FRONTEND_PUBLIC_DIR, "model")
FRONTEND_MEDIAPIPE_DIR = os.path.join(FRONTEND_PUBLIC_DIR, "mediapipe")

def copy_with_stats(src, dst):
    if not os.path.exists(src):
        print(f"Warning: Source file {src} not found. Skipping.")
        return
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    
    size_kb = os.path.getsize(dst) / 1024
    print(f"Copied {os.path.basename(src)} -> {dst} ({size_kb:.2f} KB)")

def main():
    print("Copying model and assets to frontend...")
    
    # 1. Model Files
    model_files = [
        "mudra.onnx",
        "class_names.json",
        "model_meta.json",
        "parity_samples.json"
    ]
    
    print("\n--- Model Files ---")
    for f in model_files:
        src = os.path.join(MODEL_DIR, f)
        dst = os.path.join(FRONTEND_MODEL_DIR, f)
        copy_with_stats(src, dst)
        
    print("\n--- Unit Cases ---")
    unit_src = os.path.join(OUTPUTS_DIR, "parity_unit_cases.json")
    unit_dst = os.path.join(FRONTEND_MODEL_DIR, "parity_unit_cases.json")
    copy_with_stats(unit_src, unit_dst)
        
    # 2. MediaPipe Assets
    print("\n--- MediaPipe Assets ---")
    hand_model_src = os.path.join(ASSETS_DIR, "hand_landmarker.task")
    hand_model_dst = os.path.join(FRONTEND_MEDIAPIPE_DIR, "hand_landmarker.task")
    copy_with_stats(hand_model_src, hand_model_dst)
    
    print("\nDone! The frontend is now ready to use the exported model.")

if __name__ == "__main__":
    main()
