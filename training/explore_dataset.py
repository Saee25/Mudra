import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse

from hand_landmarks import create_image_landmarker
import mediapipe as mp

"""
KNOWN DATASET LIMITATIONS (ISL Dataset)
1. Similar Backgrounds: Often recorded by one or few signers in a static environment.
   Image-based models (CNNs) might memorize the background instead of the hands.
   Our main Mudra model mitigates this by exclusively using hand landmarks (which ignore background).
   Section 10's Collect mode will add new signers to evaluate and improve generalization.
2. Consecutive Frames: Extracted from video. Randomly splitting these frames leaks near-duplicates 
   into the validation set, creating artificially high accuracy.
   We will use block-based splitting in the training step to ensure clean evaluation.
3. Two-Handed Poses: ISL contains many signs where hands overlap or touch.
   MediaPipe might struggle to consistently detect both hands in these specific frames.
   We evaluate this strictly in the Detection Gate below.
"""

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data" / "isl_alphabet"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

# Mudra Design System Colors for plots
COLOR_CREAM = "#FBF7F0"
COLOR_BLUSH = "#DA2456"
COLOR_INK = "#2E2226"

def setup_plot_style():
    plt.rcParams['figure.facecolor'] = COLOR_CREAM
    plt.rcParams['axes.facecolor'] = COLOR_CREAM
    plt.rcParams['text.color'] = COLOR_INK
    plt.rcParams['axes.labelcolor'] = COLOR_INK
    plt.rcParams['xtick.color'] = COLOR_INK
    plt.rcParams['ytick.color'] = COLOR_INK
    plt.rcParams['axes.edgecolor'] = COLOR_INK
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def get_classes_and_files():
    if not DATA_DIR.exists():
        print(f"ERROR: Dataset not found at {DATA_DIR}. Run download_dataset.py first.")
        return None
    
    classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    # Sorted order: 1-9 first, then A-Z
    classes = sorted(classes)
    
    class_files = {}
    for c in classes:
        files = list((DATA_DIR / c).glob("*.jpg")) + list((DATA_DIR / c).glob("*.png"))
        files.sort(key=lambda x: natural_sort_key(x.name))
        class_files[c] = files
        
    return class_files

def explore_basic_stats(class_files):
    print("\n--- Basic Statistics ---")
    total_images = sum(len(f) for f in class_files.values())
    print(f"Total Classes: {len(class_files)}")
    print(f"Sorted Class List: {', '.join(class_files.keys())}")
    print(f"Total Images: {total_images}")
    
    # Check a random sample for image dimensions
    sample_class = list(class_files.keys())[0]
    sample_file = class_files[sample_class][0]
    
    with Image.open(sample_file) as img:
        print(f"Sample Image Dimensions: {img.size}")
        print(f"Sample Color Mode: {img.mode}")
        
    # Check if sizes are identical across a subset
    sizes = set()
    for c, files in class_files.items():
        if files:
            with Image.open(random.choice(files)) as img:
                sizes.add(img.size)
    
    if len(sizes) == 1:
        print("Image Size Consistency: ALL sampled images are identical in size.")
    else:
        print(f"Image Size Consistency: VARIES. Found sizes: {sizes}")

def generate_class_distribution(class_files):
    counts = {c: len(f) for c, f in class_files.items()}
    
    setup_plot_style()
    plt.figure(figsize=(12, 5))
    plt.bar(counts.keys(), counts.values(), color=COLOR_BLUSH)
    plt.title("Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved class distribution plot to outputs/class_distribution.png")

def generate_sample_grid(class_files):
    setup_plot_style()
    num_classes = len(class_files)
    cols = 5
    rows = num_classes
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for row, (c, files) in enumerate(class_files.items()):
        samples = random.sample(files, min(cols, len(files)))
        for col, file in enumerate(samples):
            ax = axes[row, col]
            img = Image.open(file)
            ax.imshow(img)
            ax.axis('off')
            if col == 0:
                ax.set_title(c, loc='left', color=COLOR_INK, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "sample_grid.png", dpi=150)
    plt.close()
    print(f"Saved sample grid to outputs/sample_grid.png")

def check_near_duplicates(class_files):
    print("\n--- Consecutive Frames Check ---")
    c = list(class_files.keys())[0]
    files = class_files[c]
    
    if len(files) < 6:
        print("Not enough files to check consecutive frames.")
        return
        
    # Save 5 consecutive frames
    setup_plot_style()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img = Image.open(files[i])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "consecutive_frames.png", dpi=150)
    plt.close()
    
    # Compute similarity (Mean Absolute Difference on tiny grayscale thumbnails)
    def get_thumb(path):
        with Image.open(path) as img:
            return np.array(img.convert('L').resize((32, 32)))

    consecutive_diffs = []
    for i in range(len(files) - 1):
        if i > 100: break # Check first 100
        t1 = get_thumb(files[i])
        t2 = get_thumb(files[i+1])
        consecutive_diffs.append(np.mean(np.abs(t1.astype(float) - t2.astype(float))))
        
    random_diffs = []
    for i in range(100):
        f1, f2 = random.sample(files, 2)
        t1 = get_thumb(f1)
        t2 = get_thumb(f2)
        random_diffs.append(np.mean(np.abs(t1.astype(float) - t2.astype(float))))
        
    avg_cons = np.mean(consecutive_diffs)
    avg_rand = np.mean(random_diffs)
    
    print(f"Average Pixel Difference (Consecutive pairs): {avg_cons:.2f}")
    print(f"Average Pixel Difference (Random pairs): {avg_rand:.2f}")
    if avg_cons < (avg_rand * 0.5):
        print("CONCLUSION: Neighboring files ARE near-duplicates (likely consecutive video frames). Block split is required.")
    else:
        print("CONCLUSION: Neighboring files do not seem strongly correlated. Block split might not be strictly necessary, but remains safe.")

def detection_gate(class_files, upscale=False):
    print(f"\n--- MediaPipe Detection Gate {'(Upscaled)' if upscale else ''} ---")
    landmarker = create_image_landmarker(num_hands=2)
    
    results = []
    sample_size = 30
    
    for c, files in class_files.items():
        samples = random.sample(files, min(sample_size, len(files)))
        
        hands_0 = 0
        hands_1 = 0
        hands_2 = 0
        
        for file in samples:
            image = cv2.imread(str(file))
            if image is None: continue
            
            if upscale:
                h, w = image.shape[:2]
                if min(h, w) < 256:
                    scale = 256 / min(h, w)
                    image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            
            # MediaPipe expects RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            detection_result = landmarker.detect(mp_image)
            num_detected = len(detection_result.hand_landmarks)
            
            if num_detected == 0: hands_0 += 1
            elif num_detected == 1: hands_1 += 1
            else: hands_2 += 1
            
        total = hands_0 + hands_1 + hands_2
        if total > 0:
            results.append({
                'class': c,
                '0_hands': hands_0 / total,
                '1_hand': hands_1 / total,
                '2_hands': hands_2 / total,
                'at_least_1': (hands_1 + hands_2) / total
            })
            
    df = pd.DataFrame(results)
    
    # Save CSVs
    suffix = "_upscaled" if upscale else ""
    df.to_csv(OUTPUTS_DIR / f"detection_gate{suffix}.csv", index=False)
    
    two_hand_classes = df[df['2_hands'] >= 0.5]['class'].tolist()
    pd.DataFrame({'class': two_hand_classes}).to_csv(OUTPUTS_DIR / f"two_hand_classes{suffix}.csv", index=False)
    
    # Verdict logic
    overall_at_least_1 = df['at_least_1'].mean()
    min_class_at_least_1 = df['at_least_1'].min()
    worst_classes = df.nsmallest(3, 'at_least_1')
    
    print(f"Overall % with >= 1 hand detected: {overall_at_least_1*100:.1f}%")
    print(f"Worst class %: {min_class_at_least_1*100:.1f}%")
    print("Worst 3 classes:")
    for _, row in worst_classes.iterrows():
        print(f"  {row['class']}: {row['at_least_1']*100:.1f}%")
        
    print(f"Classes mostly two-handed (>=50%): {', '.join(two_hand_classes)}")
    
    verdict = "FAIL"
    if overall_at_least_1 >= 0.85 and min_class_at_least_1 >= 0.70:
        verdict = "PASS"
    elif overall_at_least_1 >= 0.75 and min_class_at_least_1 >= 0.50:
        verdict = "WARN"
        
    print(f"\nVERDICT: {verdict}")
    
    # Plot
    setup_plot_style()
    plt.figure(figsize=(14, 5))
    x = np.arange(len(df['class']))
    width = 0.8
    plt.bar(x, df['at_least_1'], width, color=COLOR_BLUSH)
    plt.axhline(y=0.85, color=COLOR_INK, linestyle='--', alpha=0.5, label='Target (85%)')
    plt.xticks(x, df['class'])
    plt.title(f"Detection Gate: % with >= 1 Hand Detected {'(Upscaled)' if upscale else ''}")
    plt.ylabel("Percentage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"detection_gate{suffix}.png", dpi=150)
    plt.close()
    
    return verdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gate", action="store_true", help="Skip the MediaPipe detection gate")
    args = parser.parse_args()
    
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    class_files = get_classes_and_files()
    if not class_files: return
    
    explore_basic_stats(class_files)
    generate_class_distribution(class_files)
    generate_sample_grid(class_files)
    check_near_duplicates(class_files)
    
    if not args.skip_gate:
        verdict = detection_gate(class_files)
        if verdict in ["WARN", "FAIL"]:
            print("\nAttempting upscale fix...")
            new_verdict = detection_gate(class_files, upscale=True)
            if new_verdict == "FAIL":
                print("\nSTOP: Detection Gate FAILED even with upscaling.")
                print("The dataset images might be too small, cropped too tightly, or heavily preprocessed.")
                print("Consider alternative datasets or relying heavily on Collect mode data.")

if __name__ == "__main__":
    main()
