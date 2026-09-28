import os
import cv2
import numpy as np
import glob
from hand_landmarks import create_image_landmarker, build_landmark_input
import mediapipe as mp

FRAMES_PER_VIDEO = 30
DATA_DIR = "data/include" 
OUTPUT_FILE = "data/landmarks/include_landmarks.npz"

def process_video(video_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    
    frames_landmarks = []
    frames_mask = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        orig_h, orig_w = frame.shape[:2]
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        detection_result = landmarker.detect(mp_image)
        
        lm_out, mask = build_landmark_input(detection_result.hand_landmarks, orig_w, orig_h)
        frames_landmarks.append(lm_out)
        frames_mask.append(mask)
        
    cap.release()
    
    if len(frames_landmarks) == 0:
        return None, None
        
    lms = np.array(frames_landmarks, dtype=np.float32)
    masks = np.array(frames_mask, dtype=np.float32)
    
    # Pad or sample to FRAMES_PER_VIDEO
    num_frames = len(lms)
    if num_frames > FRAMES_PER_VIDEO:
        indices = np.linspace(0, num_frames - 1, FRAMES_PER_VIDEO, dtype=int)
        lms = lms[indices]
        masks = masks[indices]
    elif num_frames < FRAMES_PER_VIDEO:
        pad_len = FRAMES_PER_VIDEO - num_frames
        lms = np.concatenate([lms, np.repeat(lms[-1:], pad_len, axis=0)], axis=0)
        masks = np.concatenate([masks, np.repeat(masks[-1:], pad_len, axis=0)], axis=0)
        
    return lms, masks

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    X = []
    M = []
    Y = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist.")
        return
        
    # Discover classes by looking one level deeper: data/include/<category>/<class_name>
    class_dirs = []
    for category in os.listdir(DATA_DIR):
        cat_path = os.path.join(DATA_DIR, category)
        if os.path.isdir(cat_path):
            for class_name in os.listdir(cat_path):
                class_path = os.path.join(cat_path, class_name)
                if os.path.isdir(class_path):
                    class_dirs.append((class_name, class_path))
                    
    # Sort classes alphabetically to ensure consistent indices
    class_dirs.sort(key=lambda x: x[0])
    
    classes = [c[0] for c in class_dirs]
    
    landmarker = create_image_landmarker(num_hands=2)
    
    for class_idx, (class_name, class_path) in enumerate(class_dirs):
        video_paths = glob.glob(os.path.join(class_path, "*.mp4")) + \
                      glob.glob(os.path.join(class_path, "*.MOV")) + \
                      glob.glob(os.path.join(class_path, "*.mov"))
                      
        if not video_paths:
            continue
            
        print(f"Processing class: {class_name} ({len(video_paths)} videos)")
        
        for v_path in video_paths:
            lms, masks = process_video(v_path, landmarker)
            if lms is not None:
                X.append(lms)
                M.append(masks)
                Y.append(class_idx)
                
    if not X:
        print("No data extracted!")
        return
        
    X = np.array(X, dtype=np.float32)
    M = np.array(M, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    classes_arr = np.array(classes, dtype=str)
    
    print(f"Saving data to {OUTPUT_FILE}")
    print(f"X shape: {X.shape}, M shape: {M.shape}, Y shape: {Y.shape}")
    
    np.savez(OUTPUT_FILE, landmarks=X, hand_mask=M, labels=Y, class_names=classes_arr)

if __name__ == "__main__":
    main()
