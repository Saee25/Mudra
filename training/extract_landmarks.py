import os
import argparse
import csv
import json
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool

from hand_landmarks import (
    ensure_hand_model,
    create_image_landmarker,
    build_landmark_input,
    compute_crop_box,
    CROP_SIZE
)

"""
Output isl_landmarks.npz schema:
- landmarks: float32 [N, 2, 21, 3] processed pixel-space input for the model
- hand_mask: float32 [N, 2] flags [1, 0] or [1, 1] indicating slots used
- raw_landmarks: float32 [N, 2, 21, 3] MediaPipe original normalized coords
- raw_count: int32 [N] number of hands detected (1 or 2)
- paths: str [N] relative image path (e.g., 'A/img.jpg')
- frame_w: int32 [N] original frame width
- frame_h: int32 [N] original frame height
- handedness: str [N] JSON encoded array of MediaPipe handedness objects
- labels: str [N] class string label (e.g., 'A')
- label_idx: int32 [N] integer index mapping to class_names
- class_names: str [C] sorted array of class string names
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "isl_alphabet")
CROP_DATA_DIR = os.path.join(DATA_DIR, "isl_cropped")
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")
SHARDS_DIR = os.path.join(LANDMARKS_DIR, "shards")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")

def process_class(args):
    cls, max_limit, upscale, no_crops = args
    
    # Each worker gets its own landmarker (they must not be shared across processes)
    landmarker = create_image_landmarker(num_hands=2)
    
    cls_dir = os.path.join(RAW_DATA_DIR, cls)
    if not os.path.isdir(cls_dir):
        return None
        
    shard_path = os.path.join(SHARDS_DIR, f"{cls}.npz")
    done_path = os.path.join(SHARDS_DIR, f"{cls}.done")
    
    if os.path.exists(shard_path) and os.path.exists(done_path):
        return {"class": cls, "status": "skipped", "reason": "shard_complete"}
        
    images = [img for img in os.listdir(cls_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    
    if max_limit is not None:
        images = images[:max_limit]
        
    crop_cls_dir = os.path.join(CROP_DATA_DIR, cls)
    if not no_crops:
        os.makedirs(crop_cls_dir, exist_ok=True)
        
    skipped_paths = []
    
    landmarks_list = []
    hand_mask_list = []
    raw_landmarks_list = []
    raw_count_list = []
    paths_list = []
    frame_w_list = []
    frame_h_list = []
    handedness_list = []
    labels_list = []
    
    stats = {"total": len(images), "0_hands": 0, "1_hand": 0, "2_hands": 0}
    
    for img_name in images:
        img_path = os.path.join(cls_dir, img_name)
        rel_path = f"{cls}/{img_name}"
        
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            continue
            
        orig_h, orig_w = bgr_img.shape[:2]
        
        # Upscale helps MediaPipe find hands in small images, which are common in Kaggle datasets. 
        # Since MediaPipe landmarks are normalized, they easily map back to the original size.
        img_for_detection = bgr_img
        if upscale and min(orig_w, orig_h) < 256:
            scale = 256.0 / min(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img_for_detection = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
        # OpenCV loads BGR, MediaPipe expects RGB
        rgb_img = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        detection_result = landmarker.detect(mp_image)
        
        hands_detected = len(detection_result.hand_landmarks)
        if hands_detected == 1:
            stats["1_hand"] += 1
        else:
            stats[f"{min(hands_detected, 2)}_hands"] += 1
        
        if hands_detected == 0:
            skipped_paths.append(rel_path)
            continue
            
        # build_landmark_input maps normalized landmarks back to original width/height
        lm_out, mask = build_landmark_input(detection_result.hand_landmarks, orig_w, orig_h)
        
        # We record the MediaPipe original normalized coordinates (zero padded) for JS tests (Section 6).
        raw_lms = np.zeros((2, 21, 3), dtype=np.float32)
        for i, h_lms in enumerate(detection_result.hand_landmarks[:2]):
            for j, lm in enumerate(h_lms):
                raw_lms[i, j] = [lm.x, lm.y, lm.z]
                
        handedness_data = []
        for h in detection_result.handedness[:2]:
            cat = h[0]
            handedness_data.append({"label": cat.category_name, "score": float(cat.score)})
            
        landmarks_list.append(lm_out)
        hand_mask_list.append(mask)
        raw_landmarks_list.append(raw_lms)
        raw_count_list.append(hands_detected)
        paths_list.append(rel_path)
        frame_w_list.append(orig_w)
        frame_h_list.append(orig_h)
        handedness_list.append(json.dumps(handedness_data))
        labels_list.append(cls)
        
        # If it's a two-hand class but only 1 hand was detected, we KEEP it.
        # Why? The same miss happens live in the browser, so the model should learn to cope with it.
        
        if not no_crops:
            crop_out_path = os.path.join(crop_cls_dir, img_name)
            if not os.path.exists(crop_out_path):
                # all_landmarks_px is a flat list of original pixel points for all detected hands
                all_px = []
                for h_lms in detection_result.hand_landmarks:
                    for lm in h_lms:
                        all_px.append([lm.x * orig_w, lm.y * orig_h, lm.z * orig_w])
                        
                x0, y0, side = compute_crop_box(all_px, orig_w, orig_h)
                
                crop_img = bgr_img[y0:y0+side, x0:x0+side]
                
                if side > CROP_SIZE:
                    interp = cv2.INTER_AREA # shrinking
                else:
                    interp = cv2.INTER_CUBIC # enlarging
                    
                crop_resized = cv2.resize(crop_img, (CROP_SIZE, CROP_SIZE), interpolation=interp)
                
                # Tradeoff: JPEG quality 95 gives smaller size with minimal artifacts. 
                # Uncompressed PNG is too large, lower JPEG creates artifacts that affect learning.
                cv2.imwrite(crop_out_path, crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
    if len(landmarks_list) > 0:
        np.savez(
            shard_path,
            landmarks=np.array(landmarks_list, dtype=np.float32),
            hand_mask=np.array(hand_mask_list, dtype=np.float32),
            raw_landmarks=np.array(raw_landmarks_list, dtype=np.float32),
            raw_count=np.array(raw_count_list, dtype=np.int32),
            paths=np.array(paths_list, dtype=str),
            frame_w=np.array(frame_w_list, dtype=np.int32),
            frame_h=np.array(frame_h_list, dtype=np.int32),
            handedness=np.array(handedness_list, dtype=str),
            labels=np.array(labels_list, dtype=str)
        )
        
    with open(done_path, 'w') as f:
        f.write("done")
        
    return {"class": cls, "status": "processed", "stats": stats, "skipped": skipped_paths}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", help="Specific classes to process")
    parser.add_argument("--limit", type=int, help="Max images per class")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--no-crops", action="store_true", help="Skip cropping")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    os.makedirs(SHARDS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    if args.classes:
        classes = args.classes
    else:
        if os.path.exists(RAW_DATA_DIR):
            classes = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
        else:
            print(f"Directory {RAW_DATA_DIR} does not exist. (Did you download the dataset?)")
            return
            
    classes.sort()
    tasks = [(cls, args.limit, True, args.no_crops) for cls in classes]
    
    summary = []
    all_skipped = []
    
    if args.workers > 1:
        with Pool(args.workers) as p:
            results = list(tqdm(p.imap_unordered(process_class, tasks), total=len(tasks), desc="Processing Classes"))
    else:
        results = []
        for task in tqdm(tasks, desc="Processing Classes"):
            results.append(process_class(task))
            
    for res in results:
        if res is None or res["status"] == "skipped":
            continue
        cls = res["class"]
        st = res["stats"]
        total = st["total"]
        if total == 0:
            continue
        skip_pct = st["0_hands"] / total * 100
        summary.append({
            "class": cls,
            "total": total,
            "0_hands": st["0_hands"],
            "1_hand": st["1_hand"],
            "2_hands": st["2_hands"],
            "skip_pct": skip_pct
        })
        all_skipped.extend(res["skipped"])
        
        if skip_pct > 20:
            print(f"\nWARNING: Class '{cls}' has a high skip rate ({skip_pct:.1f}%).")
            print("Consider lower detection confidence, upscaling, or more Collect-mode data.")
            
    skipped_file = os.path.join(OUTPUTS_DIR, "skipped_images.txt")
    existing_skipped = set()
    if os.path.exists(skipped_file):
        with open(skipped_file, "r") as f:
            existing_skipped = set(line.strip() for line in f)
            
    new_skipped = set(all_skipped) - existing_skipped
    if new_skipped:
        with open(skipped_file, "a") as f:
            for p in new_skipped:
                f.write(p + "\n")
                
    print("\nMerging shards...")
    class_names = sorted(classes)
    merged_data = {
        "landmarks": [], "hand_mask": [], "raw_landmarks": [],
        "raw_count": [], "label_idx": [], "labels": [],
        "paths": [], "frame_w": [], "frame_h": [], "handedness": []
    }
    
    for cls in class_names:
        shard_path = os.path.join(SHARDS_DIR, f"{cls}.npz")
        if os.path.exists(shard_path):
            data = np.load(shard_path)
            if len(data["paths"]) > 0:
                merged_data["landmarks"].append(data["landmarks"])
                merged_data["hand_mask"].append(data["hand_mask"])
                merged_data["raw_landmarks"].append(data["raw_landmarks"])
                merged_data["raw_count"].append(data["raw_count"])
                merged_data["paths"].append(data["paths"])
                merged_data["frame_w"].append(data["frame_w"])
                merged_data["frame_h"].append(data["frame_h"])
                merged_data["handedness"].append(data["handedness"])
                
                lbls = data["labels"]
                merged_data["labels"].append(lbls)
                idx = class_names.index(cls)
                merged_data["label_idx"].append(np.full(len(lbls), idx, dtype=np.int32))
            
    if merged_data["landmarks"]:
        final_dict = {k: np.concatenate(v) for k, v in merged_data.items()}
        final_dict["class_names"] = np.array(class_names, dtype=str)
        
        out_npz = os.path.join(LANDMARKS_DIR, "isl_landmarks.npz")
        np.savez(out_npz, **final_dict)
        print(f"Saved merged landmarks to {out_npz}")
        
    if summary:
        csv_path = os.path.join(OUTPUTS_DIR, "preprocessing_summary.csv")
        # Read existing to not overwrite other classes if only processing a subset
        existing_rows = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows[row["class"]] = row
        
        for row in summary:
            row["skip_pct"] = f"{row['skip_pct']:.2f}"
            existing_rows[row["class"]] = row
            
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "total", "0_hands", "1_hand", "2_hands", "skip_pct"])
            writer.writeheader()
            for cls in sorted(existing_rows.keys()):
                writer.writerow(existing_rows[cls])
                
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            print("Generating visualization...")
            samples_to_plot = []
            for row in summary:
                cls = row["class"]
                shard_path = os.path.join(SHARDS_DIR, f"{cls}.npz")
                if os.path.exists(shard_path):
                    data = np.load(shard_path)
                    if len(data["paths"]) > 0:
                        idx = min(len(data["paths"]) - 1, 5) # grab arbitrary index
                        samples_to_plot.append({
                            "cls": cls,
                            "path": data["paths"][idx],
                            "landmarks": data["landmarks"][idx],
                            "mask": data["hand_mask"][idx],
                            "count": data["raw_count"][idx]
                        })
                if len(samples_to_plot) >= 4:
                    break
                    
            if samples_to_plot:
                fig, axs = plt.subplots(len(samples_to_plot), 2, figsize=(10, 5 * len(samples_to_plot)))
                if len(samples_to_plot) == 1:
                    axs = [axs]
                
                blush = "#DA2456"
                ink = "#2E2226"
                crop_color = "#EE456B"
                cream_bg = "#FFFDF8"
                fig.patch.set_facecolor(cream_bg)
                
                for ax_row, sample in zip(axs, samples_to_plot):
                    img_path = os.path.join(RAW_DATA_DIR, sample["path"])
                    bgr_img = cv2.imread(img_path)
                    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    
                    ax_orig, ax_crop = ax_row
                    ax_orig.imshow(rgb_img)
                    ax_orig.set_title(f"Class: {sample['cls']} | Hands: {sample['count']}", color=ink)
                    ax_orig.axis("off")
                    
                    lms = sample["landmarks"]
                    mask = sample["mask"]
                    all_px = []
                    
                    for i in range(2):
                        if mask[i] == 1:
                            color = blush if i == 0 else ink
                            xs = lms[i, :, 0]
                            ys = lms[i, :, 1]
                            ax_orig.scatter(xs, ys, c=color, s=10)
                            for x, y in zip(xs, ys):
                                all_px.append([x, y, 0])
                                
                    h, w = rgb_img.shape[:2]
                    x0, y0, side = compute_crop_box(all_px, w, h)
                    rect = plt.Rectangle((x0, y0), side, side, fill=False, edgecolor=crop_color, linewidth=2)
                    ax_orig.add_patch(rect)
                    
                    crop_path = os.path.join(CROP_DATA_DIR, sample["path"])
                    if os.path.exists(crop_path):
                        crop_bgr = cv2.imread(crop_path)
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        ax_crop.imshow(crop_rgb)
                    else:
                        ax_crop.text(0.5, 0.5, "Crop not found", ha='center')
                        
                    ax_crop.set_title("224x224 Crop", color=ink)
                    ax_crop.axis("off")
                    
                plt.tight_layout()
                vis_path = os.path.join(OUTPUTS_DIR, "landmark_examples.png")
                plt.savefig(vis_path, facecolor=fig.get_facecolor(), dpi=150)
                print(f"Visualization saved to {vis_path}")
        except ImportError:
            print("matplotlib is not installed, skipping visualization.")

if __name__ == "__main__":
    main()
