import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap

from common import get_device, LandmarkDataset, CropImageDataset
from model_defs import LandmarkMLP, MudraMobileNetV2, count_parameters

# Colormap: #FFFDF8 -> #FAA7B6 -> #84173F
MUDRA_CMAP = LinearSegmentedColormap.from_list("mudra", ["#FFFDF8", "#FAA7B6", "#84173F"])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["landmark", "image", "both"], required=True)
    parser.add_argument("--test-set", type=str, default="val")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    return parser.parse_args()

def evaluate_model(args, model_type):
    print(f"\n--- Evaluating {model_type} model on {args.test_set} set ---")
    if args.test_set == "collected":
        print("Evaluating on collected holdout set.")
        
    outputs_dir = args.outputs_dir
    with open(os.path.join(outputs_dir, "class_names.json"), "r") as f:
        class_names = json.load(f)
        
    split_filename = "split.json" if args.test_set != "collected" else "collected_split.json"
    with open(os.path.join(outputs_dir, split_filename), "r") as f:
        split_data = json.load(f)
    test_key = args.test_set if args.test_set != "collected" else "test_indices"
    test_paths = set(split_data[test_key])
    
    device = get_device()
    is_landmark = (model_type == "landmark")
    
    if is_landmark:
        if args.test_set == "collected":
            npz_path = os.path.join(args.data_dir, "landmarks", "collected_landmarks.npz")
            from common import CollectedLandmarkDataset
            dataset = CollectedLandmarkDataset(npz_path, list(test_paths), class_names, is_train=False)
        else:
            npz_path = os.path.join(args.data_dir, "landmarks", "isl_landmarks.npz")
            dataset = LandmarkDataset(npz_path, test_paths, class_names, is_train=False)
        model = LandmarkMLP(len(class_names))
        ckpt_path = os.path.join(args.models_dir, "landmark_mlp.pt")
        prefix = "landmark_mlp"
    else:
        if args.test_set == "collected":
            npz_path = os.path.join(args.data_dir, "landmarks", "collected_landmarks.npz")
            data = np.load(npz_path)
            sessions = data["session_id"]
            test_sessions = set([sessions[i] for i in test_paths])
            
            allowed_paths = set()
            crop_dir = os.path.join(args.data_dir, "collected_crops")
            if os.path.exists(crop_dir):
                import glob
                for label in class_names:
                    if label == "other": continue
                    pattern = os.path.join(crop_dir, label, "*.jpg")
                    for p in glob.glob(pattern):
                        basename = os.path.basename(p)
                        for s in test_sessions:
                            if basename.startswith(s + "_"):
                                allowed_paths.add(f"{label}/{basename}")
                                break
            dataset = CropImageDataset(crop_dir, allowed_paths, class_names, is_train=False)
        else:
            crop_dir = os.path.join(args.data_dir, "isl_cropped")
            dataset = CropImageDataset(crop_dir, test_paths, class_names, is_train=False)
        model = MudraMobileNetV2(len(class_names))
        ckpt_path = os.path.join(args.models_dir, "image_cnn.pt")
        prefix = "image_cnn"
        
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Skipping.")
        return
        
    if len(dataset) == 0:
        print(f"Dataset for {model_type} is empty on {args.test_set} set. Skipping.")
        return
        
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint.get("class_names") and checkpoint["class_names"] != class_names:
        raise ValueError("Checkpoint class names do not match class_names.json!")
        
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    all_masks = []
    
    from torch.utils.data import DataLoader
    batch_size = 64 if not is_landmark else 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("Collecting predictions...")
    with torch.inference_mode():
        for batch in dataloader:
            if is_landmark:
                lms, mask, labels = batch
                lms, mask = lms.to(device), mask.to(device)
                out = model(lms, mask)
                all_masks.extend(mask.cpu().numpy())
            else:
                imgs, labels = batch
                imgs = imgs.to(device)
                out = model(imgs)
                
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    if is_landmark:
        all_masks = np.array(all_masks)
        
    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Overall Accuracy: {acc:.4f}")
    
    # Classification    # Generate metrics
    cr = classification_report(all_labels, all_preds, labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
    cr_df = pd.DataFrame(cr).transpose()
    cr_csv_path = os.path.join(outputs_dir, f"{prefix}_{args.test_set}_classification_report.csv")
    cr_df.to_csv(cr_csv_path)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    cax = ax.matshow(cm_norm * 100, cmap=MUDRA_CMAP)
    plt.colorbar(cax, fraction=0.046, pad=0.04, label='Percentage')
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)
    
    # Move x-axis labels to bottom
    ax.xaxis.set_ticks_position('bottom')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type.capitalize()} Model Confusion Matrix ({args.test_set} set)')
    plt.savefig(os.path.join(outputs_dir, f"{prefix}_{args.test_set}_confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    
    # Top confusions
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pct = (cm[i, j] / cm[i].sum()) * 100 if cm[i].sum() > 0 else 0
                confusions.append({
                    "true": class_names[i], "pred": class_names[j], 
                    "pct": pct, "count": cm[i, j], "total": cm[i].sum()
                })
    confusions.sort(key=lambda x: x["pct"], reverse=True)
    
    top_c_path = os.path.join(outputs_dir, f"{prefix}_{args.test_set}_top_confusions.txt")
    with open(top_c_path, "w") as f:
        f.write("Top Confusions:\n")
        for c in confusions[:15]:
            f.write(f"{c['true']} was predicted as {c['pred']} in {c['pct']:.1f}% of {c['true']} samples ({c['count']} of {c['total']}).\n")
            
        f.write("\nGrouped Confusions:\n")
        ll, dd, ld = 0, 0, 0
        for c in confusions:
            t_is_digit = c["true"].isdigit()
            p_is_digit = c["pred"].isdigit()
            if not t_is_digit and not p_is_digit:
                ll += c["count"]
            elif t_is_digit and p_is_digit:
                dd += c["count"]
            else:
                ld += c["count"]
        f.write(f"Letter <-> Letter errors: {ll}\n")
        f.write(f"Digit <-> Digit errors: {dd}\n")
        f.write(f"Letter <-> Digit errors: {ld}\n")
        
    # Landmark specific
    if is_landmark:
        hands_detected = all_masks.sum(axis=1)
        acc_1hand = accuracy_score(all_labels[hands_detected == 1], all_preds[hands_detected == 1]) if sum(hands_detected == 1) > 0 else 0
        acc_2hand = accuracy_score(all_labels[hands_detected == 2], all_preds[hands_detected == 2]) if sum(hands_detected == 2) > 0 else 0
        
        class_avg_hands = np.zeros(len(class_names))
        for i in range(len(class_names)):
            if sum(all_labels == i) > 0:
                class_avg_hands[i] = hands_detected[all_labels == i].mean()
        two_hand_classes = np.where(class_avg_hands > 1.5)[0]
        
        mask_2hc_1h = np.isin(all_labels, two_hand_classes) & (hands_detected == 1)
        acc_missed_hand = accuracy_score(all_labels[mask_2hc_1h], all_preds[mask_2hc_1h]) if sum(mask_2hc_1h) > 0 else 0
        
        with open(os.path.join(outputs_dir, "landmark_mlp_val_by_hand_count.csv"), "w") as f:
            f.write("condition,accuracy,support\n")
            f.write(f"1_hand_detected,{acc_1hand:.4f},{sum(hands_detected == 1)}\n")
            f.write(f"2_hands_detected,{acc_2hand:.4f},{sum(hands_detected == 2)}\n")
            f.write(f"two_hand_class_with_1_hand_detected,{acc_missed_hand:.4f},{sum(mask_2hc_1h)}\n")
            
        print("Running mirror check...")
        mirror_preds = []
        with torch.inference_mode():
            for i in range(len(dataset)):
                lms = dataset.landmarks[i].copy()
                mask = dataset.hand_mask[i].copy()
                w = dataset.frame_w[i]
                lms[..., 0] = w - lms[..., 0]
                if mask[0] == 1 and mask[1] == 1:
                    if lms[1, 0, 0] < lms[0, 0, 0]:
                        lms[[0, 1]] = lms[[1, 0]]
                out = model(torch.from_numpy(lms).unsqueeze(0).to(device), torch.from_numpy(mask).unsqueeze(0).to(device))
                mirror_preds.append(out.argmax(dim=1).item())
        mirror_preds = np.array(mirror_preds)
        acc_mirror = accuracy_score(all_labels, mirror_preds)
        print(f"Mirrored Accuracy: {acc_mirror:.4f}")
        
        cm_m = confusion_matrix(all_labels, mirror_preds, labels=np.arange(len(class_names)))
        sharp_rises = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j:
                    diff = cm_m[i, j] - cm[i, j]
                    if diff > max(5, 0.05 * cm[i].sum()):
                        sharp_rises.append((class_names[i], class_names[j], cm[i, j], cm_m[i, j]))
        if sharp_rises:
            print("\nWARNING: Some class pairs show a sharp rise in confusion when mirrored!")
            print("These might be mirror images of each other. Consider checking MIRROR_AUGMENT.")
            for sr in sharp_rises:
                print(f"  {sr[0]} -> {sr[1]}: Normal {sr[2]}, Mirrored {sr[3]}")
                
    # Training curves
    hist_path = os.path.join(outputs_dir, f"{prefix}_history.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#FFFDF8')
        
        ax1.plot(hist['epoch'], hist['train_acc'], color='#84173F', label='Train')
        if 'val_acc' in hist.columns:
            ax1.plot(hist['epoch'], hist['val_acc'], color='#FAA7B6', label='Val')
        ax1.set_title('Accuracy')
        ax1.set_facecolor('#FFFDF8')
        ax1.legend()
        
        ax2.plot(hist['epoch'], hist['train_loss'], color='#84173F', label='Train')
        if 'val_loss' in hist.columns:
            ax2.plot(hist['epoch'], hist['val_loss'], color='#FAA7B6', label='Val')
        ax2.set_title('Loss')
        ax2.set_facecolor('#FFFDF8')
        ax2.legend()
        
        if not is_landmark and 'phase' in hist.columns:
            phase2_starts = hist[hist['phase'] == 'phase2']['epoch']
            if len(phase2_starts) > 0:
                p2_start = phase2_starts.iloc[0]
                ax1.axvline(x=p2_start, color='#2E2226', linestyle='--', alpha=0.5)
                ax2.axvline(x=p2_start, color='#2E2226', linestyle='--', alpha=0.5)
                
        plt.savefig(os.path.join(outputs_dir, f"{prefix}_accuracy_curve.png"), bbox_inches='tight', facecolor='#FFFDF8')
        plt.close()
        
    # Latency (CPU)
    print("Measuring latency (CPU)...")
    model.cpu()
    sample = dataset[0]
    with torch.inference_mode():
        if is_landmark:
            inp = (sample[0].unsqueeze(0).cpu(), sample[1].unsqueeze(0).cpu())
        else:
            inp = (sample[0].unsqueeze(0).cpu(),)
            
        for _ in range(10):
            model(*inp)
            
        times = []
        for i in range(min(500, len(dataset))):
            s = dataset[i]
            if is_landmark:
                inp = (s[0].unsqueeze(0).cpu(), s[1].unsqueeze(0).cpu())
            else:
                inp = (s[0].unsqueeze(0).cpu(),)
            t0 = time.perf_counter()
            model(*inp)
            times.append(time.perf_counter() - t0)
            
    latency_ms = np.mean(times) * 1000
    
    total_params, _ = count_parameters(model)
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    
    metrics = {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "parameters": total_params,
        "size_mb": ckpt_size_mb,
        "latency_ms": latency_ms
    }
    
    with open(os.path.join(outputs_dir, f"{prefix}_{args.test_set}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def compare_models(args):
    out = args.outputs_dir
    lm_path = os.path.join(out, f"landmark_mlp_{args.test_set}_metrics.json")
    im_path = os.path.join(out, f"image_cnn_{args.test_set}_metrics.json")
    
    if os.path.exists(lm_path) and os.path.exists(im_path):
        with open(lm_path, "r") as f: lm = json.load(f)
        with open(im_path, "r") as f: im = json.load(f)
        
        df = pd.DataFrame([
            {"model": "landmark_mlp", "input": "landmarks", **lm},
            {"model": "image_cnn", "input": "pixels", **im}
        ])
        df.to_csv(os.path.join(out, "model_comparison.csv"), index=False)
        
        size_ratio = im['size_mb'] / lm['size_mb'] if lm['size_mb'] > 0 else 0
        lat_ratio = im['latency_ms'] / lm['latency_ms'] if lm['latency_ms'] > 0 else 0
        acc_diff = (lm['accuracy'] - im['accuracy']) * 100
        
        print("\n=== MODEL COMPARISON ===")
        print(df.to_string())
        
        # Read the test set string
        is_collected = (args.test_set == "collected")
        
        print(f"\nThe landmark MLP is {size_ratio:.1f}x smaller and {lat_ratio:.1f}x faster than MobileNetV2.")
        if is_collected:
            print(f"On an unseen signer, the landmark MLP scores {lm['accuracy']*100:.1f}% vs MobileNetV2's {im['accuracy']*100:.1f}%.")
        else:
            print(f"Accuracy difference of {acc_diff:+.1f} percentage points on the block-split validation set.")
            print("Note: The most meaningful comparison is on UNSEEN-SIGNER data (Section 10), "
                  "where landmark models usually hold up better because they ignore background and skin tone.")
    else:
        print("\nCannot compare: both metrics files must exist.")

if __name__ == "__main__":
    args = parse_args()
    if args.model in ["landmark", "both"]:
        evaluate_model(args, "landmark")
    if args.model in ["image", "both"]:
        evaluate_model(args, "image")
        
    if args.model == "both":
        compare_models(args)
