import os
import urllib.request
import json
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

"""
PARITY-CRITICAL FILE
This file is parity-critical. The setup and configurations here (especially the 
.task file, num_hands, and confidence values) MUST exactly match what the browser 
uses in JavaScript. This ensures the model receives identical features during 
training and live inference.
"""

# Paths relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
HAND_MODEL_PATH = os.path.join(ASSETS_DIR, "hand_landmarker.task")
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

CROP_PADDING = 0.2
CROP_SIZE = 224

def ensure_hand_model():
    """Downloads Google's official hand_landmarker.task if missing."""
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    if not os.path.exists(HAND_MODEL_PATH):
        print(f"Downloading hand_landmarker.task from {HAND_MODEL_URL}...")
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
            size_mb = os.path.getsize(HAND_MODEL_PATH) / (1024 * 1024)
            print(f"Download complete: {size_mb:.2f} MB")
        except Exception as e:
            print(f"ERROR: Failed to download MediaPipe model. {e}")
            print("Please check your internet connection or download it manually.")
            raise

def create_image_landmarker(num_hands=2, min_detection_confidence=0.3, min_presence_confidence=0.3):
    """
    Creates a MediaPipe HandLandmarker in IMAGE mode.
    
    Why these confidence values? 
    A lower threshold (~0.3-0.5) finds more hands in awkward two-hand poses 
    (where hands overlap or move fast) but increases the risk of false positives. 
    A higher threshold (e.g., 0.7) is stricter but might drop hands entirely. 
    We default to 0.3 to maximize 2-hand detection recall.
    """
    ensure_hand_model()
    
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence
    )
    
    return vision.HandLandmarker.create_from_options(options)

def build_landmark_input(hand_landmarks_list, frame_width, frame_height):
    """
    Algorithm (Parity-Critical):
    1. For each detected hand (MediaPipe returns at most 2 with num_hands=2), 
       convert normalized coordinates to pixels:
       x_px = x * frame_width
       y_px = y * frame_height
       z_px = z * frame_width
       (MediaPipe's z uses the width for its scale, so we scale by width).
    2. Sort the detected hands by wrist x_px (landmark 0), ascending.
       Tie-break: by wrist y_px, ascending. The hand at the smaller x goes in slot 0.
       Why position-based? We use position-based slots instead of MediaPipe's Left/Right 
       handedness label because handedness can flip when hands overlap or the image is 
       mirrored. Position is simple and perfectly reproducible in both languages.
    3. If one hand is detected: it goes in slot 0, slot 1 is all zeros, and mask = [1, 0]. 
       With two hands: mask = [1, 1]. With none: the caller skips the sample.
    """
    landmarks_out = np.zeros((2, 21, 3), dtype=np.float32)
    hand_mask = np.zeros(2, dtype=np.float32)
    
    if not hand_landmarks_list:
        return landmarks_out, hand_mask
        
    hands_px = []
    for hand_lms in hand_landmarks_list:
        hand_px = []
        for lm in hand_lms:
            hand_px.append([
                lm.x * frame_width,
                lm.y * frame_height,
                lm.z * frame_width
            ])
        hands_px.append(hand_px)
        
    # Sort hands by wrist (landmark 0) x_px, then y_px
    hands_px.sort(key=lambda h: (h[0][0], h[0][1]))
    
    for i in range(min(len(hands_px), 2)):
        landmarks_out[i] = hands_px[i]
        hand_mask[i] = 1.0
        
    return landmarks_out, hand_mask

def compute_crop_box(all_landmarks_px, frame_width, frame_height):
    """
    This is the square crop around the UNION of all detected hands' landmarks.
    
    Algorithm (Parity-Critical):
    1. tight box around all landmark points (both hands).
    2. side = max(box_w, box_h) * (1 + 2 * CROP_PADDING), centered on the box center.
    3. if side > min(frame_width, frame_height), set side = min(frame_width, frame_height).
    4. x0 = cx - side/2 and y0 = cy - side/2, then SHIFT the box back inside the frame 
       if it overflows an edge (keeping its size). This avoids distortion and black padding.
    5. rounding rule: round side, x0, and y0 to integers with floor(v + 0.5) 
       (explicitly NOT Python's banker's rounding, so JavaScript's Math.floor(v + 0.5) 
       matches exactly). Round side first, then compute and round x0/y0, then shift.
    """
    if not all_landmarks_px:
        return 0, 0, min(frame_width, frame_height)
        
    xs = [pt[0] for pt in all_landmarks_px]
    ys = [pt[1] for pt in all_landmarks_px]
        
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    box_w = max_x - min_x
    box_h = max_y - min_y
    cx = min_x + box_w / 2.0
    cy = min_y + box_h / 2.0
    
    side = max(box_w, box_h) * (1.0 + 2.0 * CROP_PADDING)
    
    # Cap side to max possible square crop
    max_side = min(frame_width, frame_height)
    if side > max_side:
        side = max_side
        
    # Round side first
    side = math.floor(side + 0.5)
    
    # Compute corner and round
    x0 = math.floor(cx - side / 2.0 + 0.5)
    y0 = math.floor(cy - side / 2.0 + 0.5)
    
    # Shift to keep inside frame
    if x0 < 0:
        x0 = 0
    elif x0 + side > frame_width:
        x0 = frame_width - side
        
    if y0 < 0:
        y0 = 0
    elif y0 + side > frame_height:
        y0 = frame_height - side
        
    return int(x0), int(y0), int(side)

if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        print("Running self-test for preprocessing parity...")
        
        class MockLM:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                
        test_cases = {}
        
        # 1. One hand
        h1 = [MockLM(0.1 + i*0.001, 0.2, 0.3) for i in range(21)]
        lm_out, mask = build_landmark_input([h1], 1000, 1000)
        test_cases["one_hand"] = {
            "landmarks": lm_out.tolist(),
            "mask": mask.tolist()
        }
        
        # 2. Two hands given in the "wrong" order
        h2 = [MockLM(0.05 + i*0.001, 0.2, 0.3) for i in range(21)]
        lm_out, mask = build_landmark_input([h1, h2], 1000, 1000)
        test_cases["two_hands_order"] = {
            "landmarks": lm_out.tolist(),
            "mask": mask.tolist()
        }
        
        # 3. Box near each edge (needs shift)
        pts_edge = [[5, 5, 0], [15, 15, 0]]
        x0, y0, side = compute_crop_box(pts_edge, 100, 100)
        test_cases["box_edge"] = {"x0": x0, "y0": y0, "side": side}
        
        # 4. Box larger than the frame
        pts_large = [[10, 10, 0], [90, 90, 0]]
        x0, y0, side = compute_crop_box(pts_large, 50, 50)
        test_cases["box_large"] = {"x0": x0, "y0": y0, "side": side}
        
        # 5. Exact .5 rounding cases
        pts_round = [[0, 0, 0], [9, 9, 0]]
        x0, y0, side = compute_crop_box(pts_round, 100, 100)
        test_cases["box_round"] = {"x0": x0, "y0": y0, "side": side}
        
        out_path = os.path.join(SCRIPT_DIR, "outputs", "parity_unit_cases.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(test_cases, f, indent=2)
            
        print(f"Self-test complete. Output saved to {out_path}")
        for k, v in test_cases.items():
            if "landmarks" in v:
                print(f"  {k}: mask={v['mask']}, hand0 wrist x={v['landmarks'][0][0][0]}")
            else:
                print(f"  {k}: x0={v['x0']}, y0={v['y0']}, side={v['side']}")
