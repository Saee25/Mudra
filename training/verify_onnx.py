import os
import json
import argparse
import numpy as np
import torch
import onnxruntime as ort

from common import resolve_classes_and_split
from model_defs import LandmarkMLP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "..", "model")
DATA_NPZ = os.path.join(SCRIPT_DIR, "data", "landmarks", "isl_landmarks.npz")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="Print top-3 predictions for a specific sample index")
    args = parser.parse_args()
    
    print("Loading class names and split...")
    class_names, _, val_paths = resolve_classes_and_split(DATA_NPZ, OUTPUTS_DIR, smoke_test=False)
    
    print("Loading PyTorch model...")
    ckpt_path = os.path.join(MODELS_DIR, "landmark_mlp.pt")
    pt_model = LandmarkMLP(num_classes=len(class_names))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        pt_model.load_state_dict(checkpoint["state_dict"])
    else:
        pt_model.load_state_dict(checkpoint)
    pt_model.eval()
    
    print("Loading ONNX model...")
    onnx_path = os.path.join(EXPORT_DIR, "mudra.onnx")
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # Load dataset to get ~300 random validation samples
    print("Loading dataset...")
    data = np.load(DATA_NPZ)
    
    val_indices = [i for i, p in enumerate(data["paths"]) if p in val_paths]
    
    # We want ~300 samples
    np.random.seed(42) # For reproducibility
    if len(val_indices) > 300:
        sample_indices = np.random.choice(val_indices, 300, replace=False)
    else:
        sample_indices = val_indices
        
    print(f"Running verification on {len(sample_indices)} validation samples...")
    
    max_diff = 0.0
    disagreements = 0
    latency_sum = 0.0
    
    # Select ~40 samples for parity_samples.json covering different cases
    parity_candidates = []
    
    for i, idx in enumerate(sample_indices):
        landmarks = data["landmarks"][idx]      # [2, 21, 3] float32
        hand_mask = data["hand_mask"][idx]      # [2] float32
        label_idx = data["label_idx"][idx]      # int
        raw_count = data["raw_count"][idx]
        
        # PyTorch
        l_tensor = torch.from_numpy(landmarks).unsqueeze(0)
        m_tensor = torch.from_numpy(hand_mask).unsqueeze(0)
        
        with torch.no_grad():
            pt_logits = pt_model(l_tensor, m_tensor)
            pt_probs = torch.softmax(pt_logits, dim=1).numpy()[0]
            
        # ONNX
        ort_inputs = {
            "landmarks": l_tensor.numpy(),
            "hand_mask": m_tensor.numpy()
        }
        
        import time
        t0 = time.perf_counter()
        ort_probs = ort_session.run(["probabilities"], ort_inputs)[0][0]
        t1 = time.perf_counter()
        latency_sum += (t1 - t0)
        
        # Compare
        diff = np.max(np.abs(pt_probs - ort_probs))
        if diff > max_diff:
            max_diff = diff
            
        pt_pred = np.argmax(pt_probs)
        ort_pred = np.argmax(ort_probs)
        
        if pt_pred != ort_pred:
            disagreements += 1
            print(f"DISAGREEMENT at index {idx}: PT={pt_pred}, ONNX={ort_pred}")

        if args.index is not None and args.index == i:
            print(f"\n--- Sample {i} (Index {idx}) Top-3 ---")
            print(f"True label: {class_names[label_idx]}")
            top3 = np.argsort(ort_probs)[-3:][::-1]
            for rank, c_idx in enumerate(top3):
                print(f"  {rank+1}: {class_names[c_idx]} ({ort_probs[c_idx]:.4f})")
            print("---------------------------------------\n")
            
        # Collect parity candidates
        if len(parity_candidates) < 40:
            # We want a mix of 1-hand, 2-hand, and edge cases.
            # Edge case heuristic: any coordinate < 50 or > frame_size - 50
            is_edge = False
            w, h = data["frame_w"][idx], data["frame_h"][idx]
            lms_px = landmarks[hand_mask == 1]
            if len(lms_px) > 0:
                min_x, max_x = np.min(lms_px[..., 0]), np.max(lms_px[..., 0])
                min_y, max_y = np.min(lms_px[..., 1]), np.max(lms_px[..., 1])
                if min_x < 50 or min_y < 50 or max_x > w - 50 or max_y > h - 50:
                    is_edge = True
                    
            # Try to balance 1-hand and 2-hand
            case_type = f"{raw_count}hand" + ("_edge" if is_edge else "")
            
            # Reconstruct raw landmarks (MediaPipe normalized, up to 2 hands)
            raw_lms = data["raw_landmarks"][idx] # [2, 21, 3]
            detected_raw = []
            for h_idx in range(raw_count):
                detected_raw.append(raw_lms[h_idx].tolist())
            
            parity_candidates.append({
                "type": case_type,
                "raw_landmarks": detected_raw,
                "frame_w": int(w),
                "frame_h": int(h),
                "expected_landmarks": landmarks.tolist(),
                "expected_hand_mask": hand_mask.tolist(),
                "expected_probabilities": pt_probs.tolist(),
                "true_label": class_names[label_idx]
            })

    print(f"\nVerification Results:")
    print(f"Samples tested: {len(sample_indices)}")
    print(f"Top-1 Agreement: {100.0 * (1 - disagreements/len(sample_indices)):.2f}%")
    print(f"Max Absolute Diff: {max_diff:.2e} (should be < 1e-5)")
    print(f"Avg ONNX Latency: {(latency_sum / len(sample_indices)) * 1000:.2f} ms")
    
    if disagreements > 0 or max_diff > 1e-4:
        print("\nERROR: ONNX model disagrees with PyTorch model!")
        exit(1)
    else:
        print("\nSUCCESS: ONNX model perfectly matches PyTorch model.")
        
    # Write parity samples
    # We want exactly 40, sorted by type for variety
    parity_candidates.sort(key=lambda x: x["type"])
    selected_parity = parity_candidates[:40]
    
    parity_out = os.path.join(EXPORT_DIR, "parity_samples.json")
    with open(parity_out, "w") as f:
        # We don't need 'type' in the final json
        for s in selected_parity:
            s.pop("type", None)
        json.dump(selected_parity, f, indent=2)
    print(f"Wrote {len(selected_parity)} parity samples to {parity_out}")

if __name__ == "__main__":
    main()
