import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model_defs import LandmarkLSTM, LandmarkSequenceExportWrapper

class SequenceLandmarkDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.landmarks = torch.tensor(data['landmarks'], dtype=torch.float32)
        self.hand_mask = torch.tensor(data['hand_mask'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.class_names = data['class_names'].tolist()
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.landmarks[idx], self.hand_mask[idx], self.labels[idx]

def export_onnx(model, num_classes, class_names, output_path):
    model.eval()
    wrapper = LandmarkSequenceExportWrapper(model)
    
    dummy_landmarks = torch.randn(1, 30, 2, 21, 3)
    dummy_mask = torch.ones(1, 30, 2)
    
    torch.onnx.export(
        wrapper,
        (dummy_landmarks, dummy_mask),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["landmarks", "hand_mask"],
        output_names=["probabilities"],
        dynamic_axes={
            "landmarks": {0: "batch_size"},
            "hand_mask": {0: "batch_size"},
            "probabilities": {0: "batch_size"}
        }
    )
    
    import json
    meta_path = output_path.replace(".onnx", "_meta.json")
    meta = {
        "class_names": class_names,
        "class_count": num_classes,
        "type": "sequence"
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Exported ONNX model to {output_path}")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for lms, masks, labels in dataloader:
        lms = lms.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(lms, masks)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * lms.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    npz_path = "data/landmarks/include_landmarks.npz"
    if not os.path.exists(npz_path):
        print(f"Dataset {npz_path} not found. Run process_include.py first.")
        return
        
    dataset = SequenceLandmarkDataset(npz_path)
    
    # Train/Val split (simple 80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(dataset.class_names)
    model = LandmarkLSTM(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 50
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Eval
        model.eval()
        val_loss, val_acc = 0, 0
        total = 0
        with torch.no_grad():
            for lms, masks, labels in val_loader:
                lms = lms.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                outputs = model(lms, masks)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * lms.size(0)
                _, preds = torch.max(outputs, 1)
                val_acc += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= total
        val_acc /= total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    os.makedirs("models", exist_ok=True)
    export_path = "../frontend/public/model/mudra_phrases.onnx"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    export_onnx(model, num_classes, dataset.class_names, export_path)

if __name__ == "__main__":
    main()
