import os
import json
import random
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image

SEED = 42
VAL_FRACTION = 0.2
BLOCK_SIZE = 50
IMG_SIZE = 224
MIRROR_AUGMENT = True

COLLECTED_HOLDOUT_SIGNERS = []

def get_base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data dir")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to models dir")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to outputs dir")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny 2-batch test")
    return parser

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def setup_directories(args):
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

def resolve_classes_and_split(npz_path, outputs_dir, smoke_test=False):
    """
    Reads class names from the dataset and creates a BLOCK-BASED split.
    A random split would inflate validation accuracy because consecutive video frames 
    are near-duplicates. Grouping by blocks ensures the model is evaluated on 
    somewhat unseen variations of the sign.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found at {npz_path}")
        
    data = np.load(npz_path)
    class_names = data["class_names"].tolist()
    
    # Save or verify class_names.json
    cn_path = os.path.join(outputs_dir, "class_names.json")
    if os.path.exists(cn_path):
        with open(cn_path, "r") as f:
            existing_names = json.load(f)
        if existing_names != class_names:
            raise ValueError(f"Class names in {cn_path} differ from {npz_path}!")
    else:
        with open(cn_path, "w") as f:
            json.dump(class_names, f, indent=2)
            
    # Load paths and labels
    paths = data["paths"]
    labels = data["labels"]
    
    # Block-based split
    split_path = os.path.join(outputs_dir, "split.json")
    if not smoke_test and os.path.exists(split_path):
        print(f"Loading existing split from {split_path}")
        with open(split_path, "r") as f:
            split_data = json.load(f)
        train_paths = set(split_data["train"])
        val_paths = set(split_data["val"])
    else:
        print("Generating new block-based split...")
        train_paths = set()
        val_paths = set()
        
        # Group paths by class
        class_paths = {cls: [] for cls in class_names}
        for p, l in zip(paths, labels):
            class_paths[l].append(p)
            
        rng = random.Random(SEED)
        
        for cls, c_paths in class_paths.items():
            # Natural sort is already done by extract_landmarks.py (paths are "cls/img_name")
            c_paths.sort()
            
            blocks = [c_paths[i:i + BLOCK_SIZE] for i in range(0, len(c_paths), BLOCK_SIZE)]
            
            # Ensure at least 1 val block
            num_val = max(1, int(len(blocks) * VAL_FRACTION))
            
            rng.shuffle(blocks)
            
            val_blocks = blocks[:num_val]
            train_blocks = blocks[num_val:]
            
            for b in val_blocks:
                val_paths.update(b)
            for b in train_blocks:
                train_paths.update(b)
                
        if not smoke_test:
            with open(split_path, "w") as f:
                json.dump({"train": list(train_paths), "val": list(val_paths)}, f, indent=2)
                
    if smoke_test:
        # tiny subset for testing
        print("SMOKE TEST: using a tiny subset of data")
        t_paths = list(train_paths)[:500]
        v_paths = list(val_paths)[:100]
        return class_names, set(t_paths), set(v_paths)

    return class_names, train_paths, val_paths

def resolve_collected_split(npz_path, outputs_dir, holdout_signers=None):
    if holdout_signers is None:
        holdout_signers = set(COLLECTED_HOLDOUT_SIGNERS)
    else:
        holdout_signers = set(holdout_signers)
        
    data = np.load(npz_path)
    labels = data["labels"]
    signers = data["signer_id"]
    sessions = data["session_id"]
    
    train_indices = []
    test_indices = []
    
    unique_signers = np.unique(signers)
    
    if len(unique_signers) == 1 and not holdout_signers:
        print("WARNING: Only one signer found in collected data. Holding out by session instead. This is weaker evidence than an unseen signer.")
        test_sessions = set()
        labels_seen = set()
        for i in range(len(labels)):
            if labels[i] not in labels_seen:
                test_sessions.add(sessions[i])
                labels_seen.add(labels[i])
                
        for i in range(len(labels)):
            if sessions[i] in test_sessions:
                test_indices.append(i)
            else:
                train_indices.append(i)
    else:
        for i in range(len(labels)):
            if signers[i] in holdout_signers:
                test_indices.append(i)
            else:
                train_indices.append(i)
                
    split_path = os.path.join(outputs_dir, "collected_split.json")
    with open(split_path, "w") as f:
        json.dump({"train_indices": train_indices, "test_indices": test_indices}, f, indent=2)
        
    return train_indices, test_indices

def augment_landmarks(landmarks, mask, frame_w, frame_h, rng):
    """
    Applies augmentation to PIXEL coordinates before the model's normalization.
    """
    lms = landmarks.copy()
    m = mask.copy()
    
    # Hand Dropout (simulates MediaPipe missing a hand)
    if m[0] == 1 and m[1] == 1 and rng.random() < 0.1:
        # drop one randomly
        drop_idx = 0 if rng.random() < 0.5 else 1
        keep_idx = 1 - drop_idx
        lms[0] = lms[keep_idx]
        lms[1] = 0
        m[0] = 1
        m[1] = 0

    # Mirror
    if MIRROR_AUGMENT and rng.random() < 0.5:
        # Mirror simulates left-handed signers. x -> w - x
        # Note: if classes are mirror images (e.g. p/q in ASL, though ISL differs), this hurts.
        lms[..., 0] = frame_w - lms[..., 0]
        
        # We must RE-SORT the slots by wrist x (like build_landmark_input)
        if m[0] == 1 and m[1] == 1:
            if lms[1, 0, 0] < lms[0, 0, 0]:
                lms[[0, 1]] = lms[[1, 0]]
                
    # Coordinate noise (Gaussian, sigma ~ 1% of frame)
    noise_scale = 0.01 * ((frame_w + frame_h) / 2)
    noise = rng.normal(0, noise_scale, lms.shape).astype(np.float32)
    # Z noise is smaller
    noise[..., 2] *= 0.1 
    
    # Apply noise only to present slots
    for i in range(2):
        if m[i] == 1:
            lms[i] += noise[i]
            
    # Rotation & Aspect jitter
    if m[0] == 1 or m[1] == 1:
        # Find center of present landmarks
        active_lms = lms[m == 1].reshape(-1, 3)
        cx, cy = np.mean(active_lms[:, :2], axis=0)
        
        # Rotation
        angle = rng.uniform(-15, 15)
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Aspect jitter (scale x and y independently)
        # Note: Uniform scale and translation are omitted here because LandmarkNormalization 
        # completely removes them anyway. Independent scale (aspect jitter) is NOT removed.
        scale_x = rng.uniform(0.9, 1.1)
        scale_y = rng.uniform(0.9, 1.1)
        
        for i in range(2):
            if m[i] == 1:
                x = lms[i, :, 0] - cx
                y = lms[i, :, 1] - cy
                
                nx = (x * cos_t - y * sin_t) * scale_x
                ny = (x * sin_t + y * cos_t) * scale_y
                
                lms[i, :, 0] = nx + cx
                lms[i, :, 1] = ny + cy

    return lms, m

class LandmarkDataset(Dataset):
    def __init__(self, npz_path, allowed_paths, class_names, is_train=False):
        data = np.load(npz_path)
        all_paths = data["paths"]
        
        # Filter indices
        valid_idx = [i for i, p in enumerate(all_paths) if p in allowed_paths]
        
        self.landmarks = data["landmarks"][valid_idx]
        self.hand_mask = data["hand_mask"][valid_idx]
        self.frame_w = data["frame_w"][valid_idx]
        self.frame_h = data["frame_h"][valid_idx]
        
        # Map string labels to current class_names index
        labels_str = data["labels"][valid_idx]
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        self.labels = np.array([name_to_idx[l] for l in labels_str], dtype=np.int64)
        
        self.is_train = is_train
        self.rng = np.random.RandomState(SEED) if is_train else None
        
    def __len__(self):
        return len(self.landmarks)
        
    def __getitem__(self, idx):
        lms = self.landmarks[idx]
        mask = self.hand_mask[idx]
        
        if self.is_train:
            lms, mask = augment_landmarks(lms, mask, self.frame_w[idx], self.frame_h[idx], self.rng)
            
        return torch.from_numpy(lms), torch.from_numpy(mask), self.labels[idx]

class CollectedLandmarkDataset(Dataset):
    def __init__(self, npz_path, allowed_indices, class_names, is_train=False):
        data = np.load(npz_path)
        
        self.landmarks = data["landmarks"][allowed_indices]
        self.hand_mask = data["hand_mask"][allowed_indices]
        self.frame_w = data["frame_w"][allowed_indices]
        self.frame_h = data["frame_h"][allowed_indices]
        
        labels_str = data["labels"][allowed_indices]
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        self.labels = np.array([name_to_idx[l] for l in labels_str], dtype=np.int64)
        
        self.is_train = is_train
        self.rng = np.random.RandomState(SEED) if is_train else None
        
    def __len__(self):
        return len(self.landmarks)
        
    def __getitem__(self, idx):
        lms = self.landmarks[idx]
        mask = self.hand_mask[idx]
        
        if self.is_train:
            lms, mask = augment_landmarks(lms, mask, self.frame_w[idx], self.frame_h[idx], self.rng)
            
        return torch.from_numpy(lms), torch.from_numpy(mask), self.labels[idx]

class CropImageDataset(Dataset):
    def __init__(self, crop_dir, allowed_paths, class_names, is_train=False):
        self.crop_dir = crop_dir
        self.paths = list(allowed_paths)
        self.paths.sort() # Ensure deterministic order
        self.is_train = is_train
        
        # Extract label from path (path format: "Class/img.jpg")
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        self.labels = [name_to_idx[p.split('/')[0]] for p in self.paths]
        
        # Torchvision V2 transforms
        transforms_list = []
        if is_train:
            if MIRROR_AUGMENT:
                transforms_list.append(v2.RandomHorizontalFlip(p=0.5))
            transforms_list.extend([
                v2.RandomRotation(degrees=10),
                v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                v2.ColorJitter(brightness=0.2, contrast=0.2)
            ])
            
        # MobileNet expects float32 tensors. Our InputNormalization will do x/255 and ImageNet norm.
        # So we just output 0-255 float32 here. (v2.ToDtype(torch.float32, scale=False) handles this)
        transforms_list.extend([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=False)
        ])
        
        self.transform = v2.Compose(transforms_list)
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        img_path = os.path.join(self.crop_dir, path)
        
        # Convert to RGB (OpenCV saved as BGR, but PIL reads RGB natively, torchvision expects RGB)
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img)
        
        return tensor, self.labels[idx]

# --- Training Helpers ---

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, is_image=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        if is_image:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            args = (inputs,)
        else:
            lms, mask, labels = batch
            lms, mask, labels = lms.to(device), mask.to(device), labels.to(device)
            args = (lms, mask)
            
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(*args)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(*args)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, is_image=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        if is_image:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            args = (inputs,)
        else:
            lms, mask, labels = batch
            lms, mask, labels = lms.to(device), mask.to(device), labels.to(device)
            args = (lms, mask)
            
        outputs = model(*args)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
