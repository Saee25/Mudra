import os
import json
import base64
import glob
import numpy as np
import argparse

from common import get_base_parser
from hand_landmarks import build_landmark_input

MIN_OTHER_SAMPLES = 300

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data dir")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to outputs dir")
    return parser.parse_args()

class DotDict:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, DotDict(**v))
            elif isinstance(v, list):
                setattr(self, k, [DotDict(**i) if isinstance(i, dict) else i for i in v])
            else:
                setattr(self, k, v)

def validate_parity(sample, frame_w, frame_h):
    # Mock landmarks for python build_landmark_input
    # raw_landmarks in json: nested list [num_hands][21][x,y,z]
    
    mock_raw_hands = []
    for hand_raw in sample["raw_landmarks"]:
        mock_hand = []
        for lm in hand_raw:
            # We construct a dotdict so .x .y .z works like mediapipe results
            mock_hand.append(DotDict(x=lm["x"], y=lm["y"], z=lm.get("z", 0.0)))
        mock_raw_hands.append(mock_hand)
        
    py_landmarks, py_mask = build_landmark_input(mock_raw_hands, frame_w, frame_h)
    
    js_landmarks = np.array(sample["landmarks"], dtype=np.float32)
    js_mask = np.array(sample["hand_mask"], dtype=np.float32)
    
    diff_lms = np.abs(py_landmarks - js_landmarks).max()
    diff_mask = np.abs(py_mask - js_mask).max()
    
    if diff_lms > 1e-4 or diff_mask > 1e-4:
        return False, f"Parity mismatch: max lm diff {diff_lms:.6f}, max mask diff {diff_mask:.6f}"
        
    return True, ""

def main():
    args = parse_args()
    
    collected_dir = os.path.join(args.data_dir, "collected")
    if not os.path.exists(collected_dir):
        print(f"Directory {collected_dir} not found. Nothing to import.")
        return
        
    json_files = glob.glob(os.path.join(collected_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {collected_dir}.")
        return
        
    print(f"Found {len(json_files)} JSON files in {collected_dir}.")
    
    all_landmarks = []
    all_masks = []
    all_labels = []
    all_frame_ws = []
    all_frame_hs = []
    all_signer_ids = []
    all_session_ids = []
    
    total_samples = 0
    valid_samples = 0
    parity_failures = 0
    
    crops_dir = os.path.join(args.data_dir, "collected_crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    for fpath in json_files:
        print(f"Processing {os.path.basename(fpath)}...")
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Failed to read JSON: {e}")
            continue
            
        if data.get("schema_version") != 1:
            print(f"  Skipping: unsupported schema_version {data.get('schema_version')}")
            continue
            
        if data.get("pipeline_version") != "landmark-v1":
            print(f"  Skipping: unsupported pipeline_version {data.get('pipeline_version')}")
            continue
            
        signer_id = data.get("signer_id", "unknown")
        session_id = data.get("session_id", "unknown")
        frame_w = data.get("frame_w", 0)
        frame_h = data.get("frame_h", 0)
        samples = data.get("samples", [])
        
        for idx, sample in enumerate(samples):
            total_samples += 1
            
            # Parity check
            is_valid, err_msg = validate_parity(sample, frame_w, frame_h)
            if not is_valid:
                parity_failures += 1
                if parity_failures <= 5: # Limit spam
                    print(f"  Parity error in sample {idx}: {err_msg}")
                continue
                
            label = sample["label"]
            
            # Extract crop if present
            crop_jpeg = sample.get("crop_jpeg")
            if crop_jpeg and crop_jpeg.startswith("data:image/jpeg;base64,"):
                b64_data = crop_jpeg.split(",", 1)[1]
                img_data = base64.b64decode(b64_data)
                
                label_crop_dir = os.path.join(crops_dir, label)
                os.makedirs(label_crop_dir, exist_ok=True)
                crop_path = os.path.join(label_crop_dir, f"{session_id}_{idx}.jpg")
                
                with open(crop_path, "wb") as img_f:
                    img_f.write(img_data)
                    
            all_landmarks.append(sample["landmarks"])
            all_masks.append(sample["hand_mask"])
            all_labels.append(label)
            all_frame_ws.append(frame_w)
            all_frame_hs.append(frame_h)
            all_signer_ids.append(signer_id)
            all_session_ids.append(session_id)
            valid_samples += 1

    print(f"\nImport summary: {valid_samples}/{total_samples} valid samples imported. Parity failures: {parity_failures}")
    
    if valid_samples == 0:
        print("No valid samples imported. Exiting.")
        return
        
    # Stats
    from collections import defaultdict
    counts = defaultdict(int)
    signer_counts = defaultdict(lambda: defaultdict(int))
    for label, signer in zip(all_labels, all_signer_ids):
        counts[label] += 1
        signer_counts[signer][label] += 1
        
    print("\nSamples per signer:")
    for signer, s_counts in signer_counts.items():
        print(f"  Signer: {signer}")
        for lbl, c in s_counts.items():
            print(f"    {lbl}: {c}")
            if c < 50:
                print(f"      -> WARNING: Too few samples ({c}) for {lbl} by {signer}")
                
    # Class names handling (especially "other")
    cn_path = os.path.join(args.outputs_dir, "class_names.json")
    if os.path.exists(cn_path):
        with open(cn_path, "r") as f:
            class_names = json.load(f)
    else:
        # Default fallback, should rarely happen if pipeline is run properly
        class_names = [str(i) for i in range(1, 10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
        
    other_count = counts.get("other", 0)
    if "other" in counts:
        if other_count >= MIN_OTHER_SAMPLES:
            if "other" not in class_names:
                class_names.append("other")
                with open(cn_path, "w") as f:
                    json.dump(class_names, f, indent=2)
                print(f"\nAdded 'other' class to class_names.json with {other_count} samples.")
                print("IMPORTANT: This changes the model output dimension.")
                print("You must retrain both models, re-export to ONNX, and copy to frontend.")
        else:
            print(f"\nDropping 'other' class ({other_count} samples) as it is below MIN_OTHER_SAMPLES ({MIN_OTHER_SAMPLES}).")
            # Filter out "other"
            indices = [i for i, lbl in enumerate(all_labels) if lbl != "other"]
            all_landmarks = [all_landmarks[i] for i in indices]
            all_masks = [all_masks[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]
            all_frame_ws = [all_frame_ws[i] for i in indices]
            all_frame_hs = [all_frame_hs[i] for i in indices]
            all_signer_ids = [all_signer_ids[i] for i in indices]
            all_session_ids = [all_session_ids[i] for i in indices]

    print("\nWriting collected_landmarks.npz...")
    lms_dir = os.path.join(args.data_dir, "landmarks")
    os.makedirs(lms_dir, exist_ok=True)
    npz_path = os.path.join(lms_dir, "collected_landmarks.npz")
    
    np.savez_compressed(
        npz_path,
        landmarks=np.array(all_landmarks, dtype=np.float32),
        hand_mask=np.array(all_masks, dtype=np.float32),
        labels=np.array(all_labels, dtype=str),
        frame_w=np.array(all_frame_ws, dtype=np.float32),
        frame_h=np.array(all_frame_hs, dtype=np.float32),
        signer_id=np.array(all_signer_ids, dtype=str),
        session_id=np.array(all_session_ids, dtype=str),
        class_names=np.array(class_names, dtype=str)
    )
    print("Done!")

if __name__ == "__main__":
    main()
