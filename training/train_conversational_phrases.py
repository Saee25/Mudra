import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_defs import LandmarkNormalization

class ImprovedBiGRU(nn.Module):
    """
    Industry-standard Sequence Landmark Architecture:
    - LandmarkNormalization: invariant to camera resolution and position.
    - Bidirectional GRU: processes gesture forward and backward in time.
    - Temporal Mean + Max Pooling: captures both the trajectory trend and the peak gesture apex.
    """
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()
        self.norm = LandmarkNormalization()
        self.gru = nn.GRU(
            input_size=254,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, lms_seq, mask_seq):
        B, T = lms_seq.shape[:2]
        lms_flat = lms_seq.reshape(B * T, 2, 21, 3)
        mask_flat = mask_seq.reshape(B * T, 2)
        features = self.norm(lms_flat, mask_flat)
        features_seq = features.reshape(B, T, -1)

        gru_out, _ = self.gru(features_seq)
        mean_pool = torch.mean(gru_out, dim=1)
        max_pool = torch.max(gru_out, dim=1)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.fc(pooled)

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lms_seq, mask_seq):
        logits = self.model(lms_seq, mask_seq)
        return self.softmax(logits)

def main():
    npz_path = os.path.join(SCRIPT_DIR, "data", "landmarks", "include_landmarks.npz")
    if not os.path.exists(npz_path):
        print(f"Dataset not found at {npz_path}")
        return

    d = np.load(npz_path)
    all_lms = d["landmarks"]
    all_masks = d["hand_mask"]
    all_labels = d["labels"]
    classes = d["class_names"].tolist()

    # The 20 high-value conversational phrases & words
    targets = [
        "Hello", "Good Morning", "Good afternoon", "Good evening", "Good night",
        "Thank you", "How are you", "Alright", "Pleased",
        "I", "you", "we", "they", "good", "bad", "happy", "sad",
        "Today", "Tomorrow", "Yesterday"
    ]

    target_indices = []
    target_names = []
    for t in targets:
        for i, c in enumerate(classes):
            clean = c.split(". ")[-1].strip() if ". " in c else c
            if clean.lower() == t.lower():
                target_indices.append(i)
                target_names.append(clean)
                break

    mask = np.isin(all_labels, target_indices)
    X = all_lms[mask]
    M = all_masks[mask]
    Y_raw = all_labels[mask]

    idx_map = {orig: new for new, orig in enumerate(target_indices)}
    Y = np.array([idx_map[y] for y in Y_raw])

    print(f"Loaded {len(X)} sequence samples for {len(targets)} conversational phrases.")

    # Honest 80/20 train/validation split
    np.random.seed(42)
    torch.manual_seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    tr_idx, val_idx = indices[:split], indices[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = ImprovedBiGRU(len(targets)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X[tr_idx]), torch.tensor(M[tr_idx]), torch.tensor(Y[tr_idx], dtype=torch.long)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X[val_idx]), torch.tensor(M[val_idx]), torch.tensor(Y[val_idx], dtype=torch.long)),
        batch_size=32, shuffle=False
    )

    best_val_acc = 0.0
    best_model_state = None

    print("\nStarting training with 80/20 validation to prevent overfitting...")
    for epoch in range(1, 36):
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for lms, masks, labels in tr_loader:
            lms, masks, labels = lms.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(lms, masks)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * lms.size(0)
            preds = torch.argmax(out, dim=1)
            tr_correct += (preds == labels).sum().item()
            tr_total += labels.size(0)

        tr_acc = tr_correct / tr_total
        tr_loss /= tr_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for lms, masks, labels in val_loader:
                lms, masks, labels = lms.to(device), masks.to(device), labels.to(device)
                out = model(lms, masks)
                loss = criterion(out, labels)
                val_loss += loss.item() * lms.size(0)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if epoch % 5 == 0 or epoch == 35:
            print(f"Epoch {epoch:02d}/35 | Train: {tr_acc*100:.1f}%, Loss {tr_loss:.3f} | Val: {val_acc*100:.1f}%, Loss {val_loss:.3f} | Best Val: {best_val_acc*100:.1f}%")

    print(f"\n[SUCCESS] Peak Validation Accuracy: {best_val_acc*100:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Export to ONNX
    model.eval()
    wrapper = ExportWrapper(model)
    export_path = os.path.join(SCRIPT_DIR, "..", "frontend", "public", "model", "mudra_phrases.onnx")
    meta_path = os.path.join(SCRIPT_DIR, "..", "frontend", "public", "model", "mudra_phrases_meta.json")

    print(f"\nExporting ONNX model to {export_path}...")
    dummy_lms = torch.randn(1, 30, 2, 21, 3)
    dummy_mask = torch.ones(1, 30, 2)

    torch.onnx.export(
        wrapper,
        (dummy_lms, dummy_mask),
        export_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["landmarks", "hand_mask"],
        output_names=["probabilities"],
        dynamic_axes=None # B=1 for client-side inference
    )

    meta = {
        "class_names": target_names,
        "class_count": len(target_names),
        "type": "sequence",
        "validation_accuracy": round(best_val_acc * 100, 2)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved metadata ({len(target_names)} classes) to {meta_path}")
    print("Conversational phrases model is now trained, validated, and exported to the frontend!")

if __name__ == "__main__":
    main()

