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

class FullINCLUDEBiGRU(nn.Module):
    """
    State-of-the-Art Sequence Landmark Architecture for all 262 INCLUDE classes:
    - LandmarkNormalization: Resolution- and translation-invariant.
    - 2-layer Bidirectional GRU (Hidden Size 256): Captures temporal trajectory forward and backward.
    - Dual Temporal Pooling (Mean + Max): Preserves both average trajectory and peak gesture point.
    - Two-layer classification head with Dropout (0.35) to prevent overfitting.
    """
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        self.norm = LandmarkNormalization()
        self.gru = nn.GRU(
            input_size=254,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.35
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(384, num_classes)
        )

    def forward(self, lms_seq, mask_seq):
        B, T = lms_seq.shape[:2]
        lms_flat = lms_seq.reshape(B * T, 2, 21, 3)
        mask_flat = mask_seq.reshape(B * T, 2)
        features = self.norm(lms_flat, mask_flat)
        features_seq = features.reshape(B, T, -1)

        gru_out, _ = self.gru(features_seq) # [B, T, hidden_size * 2]
        mean_pool = torch.mean(gru_out, dim=1)
        max_pool = torch.max(gru_out, dim=1)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=-1) # [B, hidden_size * 4]
        return self.classifier(pooled)

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lms_seq, mask_seq):
        logits = self.model(lms_seq, mask_seq)
        return self.softmax(logits)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SOTA BiGRU on all 262 INCLUDE phrases")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (use 64 or 128 for GPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--data-path", type=str, default=None, help="Path to include_landmarks.npz")
    args = parser.parse_args()

    npz_path = args.data_path or os.path.join(SCRIPT_DIR, "data", "landmarks", "include_landmarks.npz")
    if not os.path.exists(npz_path):
        print(f"Error: Dataset file not found at {npz_path}")
        return

    print("Loading INCLUDE landmark dataset...")
    d = np.load(npz_path)
    X, M, Y = d["landmarks"], d["hand_mask"], d["labels"]
    raw_class_names = d["class_names"].tolist()
    num_classes = len(raw_class_names)
    clean_class_names = [c.split(". ")[-1].strip() if ". " in c else c for c in raw_class_names]

    print(f"Dataset successfully loaded: {len(X)} samples across all {num_classes} classes.")

    # Honest 80/20 train/validation split to strictly prevent overfitting
    np.random.seed(42)
    torch.manual_seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    tr_idx, val_idx = indices[:split], indices[split:]

    print(f"Split: {len(tr_idx)} training samples (80%), {len(val_idx)} validation samples (20%).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} (PyTorch {torch.__version__})")

    model = FullINCLUDEBiGRU(num_classes, hidden_size=256).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X[tr_idx]), torch.tensor(M[tr_idx]), torch.tensor(Y[tr_idx], dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X[val_idx]), torch.tensor(M[val_idx]), torch.tensor(Y[val_idx], dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda")
    )

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print("\nStarting Training (monitoring holdout validation loss & top-1/top-5 accuracy)...")
    for epoch in range(1, args.epochs + 1):
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

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        top5_correct = 0
        with torch.no_grad():
            for lms, masks, labels in val_loader:
                lms, masks, labels = lms.to(device), masks.to(device), labels.to(device)
                out = model(lms, masks)
                loss = criterion(out, labels)
                val_loss += loss.item() * lms.size(0)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == labels).sum().item()

                _, top5 = torch.topk(out, 5, dim=1)
                top5_correct += torch.sum(top5 == labels.unsqueeze(1)).item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_top5 = top5_correct / val_total
        val_loss /= val_total
        scheduler.step(val_loss)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            improved = " ⭐ (New Best)"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Acc: {tr_acc*100:.1f}%, Loss: {tr_loss:.3f} | Val Top-1: {val_acc*100:.1f}%, Val Top-5: {val_top5*100:.1f}%, Loss: {val_loss:.3f}{improved}", flush=True)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch} to prevent overfitting.")
            break

    print(f"\n==================================================")
    print(f"Best Validation Top-1 Accuracy: {best_val_acc*100:.2f}%")
    print(f"==================================================")

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
        dynamic_axes=None # Fixed batch=1 for browser inference
    )

    meta = {
        "class_names": clean_class_names,
        "class_count": len(clean_class_names),
        "type": "sequence",
        "validation_accuracy": round(best_val_acc * 100, 2)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"Saved {export_path} ({size_mb:.2f} MB)")
    print(f"Saved metadata ({len(clean_class_names)} classes) to {meta_path}")
    print("Done! The frontend is now equipped with the full 262-class phrases model.")

if __name__ == "__main__":
    main()

