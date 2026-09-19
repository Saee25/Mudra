import os
import time
import argparse
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    get_base_parser, set_seed, get_device, setup_directories,
    resolve_classes_and_split, resolve_collected_split, LandmarkDataset, 
    CollectedLandmarkDataset, train_one_epoch, 
    evaluate_epoch, EarlyStopping, MIRROR_AUGMENT
)
from model_defs import LandmarkMLP, count_parameters

# --- CONFIG ---
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4 # AdamW
LABEL_SMOOTHING = 0.05 # Softens overconfident predictions on a small, clean dataset
PATIENCE = 15
COLLECTED_WEIGHT = 3

def get_args():
    parser = get_base_parser()
    parser.add_argument("--include-collected", action="store_true", help="Include samples from Collect mode")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()

def main():
    args = get_args()
    
    if args.include_collected:
        print("Collect mode enabled: Will include collected samples in training.")
        
    set_seed()
    device = get_device()
    setup_directories(args)
    
    npz_path = os.path.join(args.data_dir, "landmarks", "isl_landmarks.npz")
    
    print("Loading split and classes...")
    class_names, train_paths, val_paths = resolve_classes_and_split(
        npz_path, args.outputs_dir, smoke_test=args.smoke_test
    )
    
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    print(f"Train paths: {len(train_paths)}, Val paths: {len(val_paths)}")
    
    print("Initializing datasets...")
    # num_workers=0 because the landmarks are already fully loaded into RAM in the Dataset
    train_dataset = LandmarkDataset(npz_path, train_paths, class_names, is_train=True)
    val_dataset = LandmarkDataset(npz_path, val_paths, class_names, is_train=False)
    
    collected_test_dataset = None
    collected_samples_count = 0
    if args.include_collected:
        collected_npz = os.path.join(args.data_dir, "landmarks", "collected_landmarks.npz")
        if os.path.exists(collected_npz):
            c_train_idx, c_test_idx = resolve_collected_split(collected_npz, args.outputs_dir)
            collected_train_dataset = CollectedLandmarkDataset(collected_npz, c_train_idx, class_names, is_train=True)
            collected_test_dataset = CollectedLandmarkDataset(collected_npz, c_test_idx, class_names, is_train=False)
            collected_samples_count = len(c_train_idx)
            
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([train_dataset] + [collected_train_dataset] * COLLECTED_WEIGHT)
            print(f"Added {collected_samples_count} collected samples to training (oversampled {COLLECTED_WEIGHT}x)")
        else:
            print(f"WARNING: {collected_npz} not found. Proceeding without collected data.")
    
    b_size = 2 if args.smoke_test else BATCH_SIZE
    
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, 
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False, 
                            num_workers=0, pin_memory=(device.type == "cuda"))
                            
    model = LandmarkMLP(num_classes).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"LandmarkMLP Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    start_epoch = 0
    last_pt_path = os.path.join(args.models_dir, "landmark_mlp_last.pt")
    
    if args.resume and os.path.exists(last_pt_path):
        print(f"Resuming from {last_pt_path}")
        checkpoint = torch.load(last_pt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")
        
    history_path = os.path.join(args.outputs_dir, "landmark_mlp_history.csv")
    write_header = not (args.resume and os.path.exists(history_path))
    
    history_file = open(history_path, mode="a" if not write_header else "w", newline="")
    history_writer = csv.writer(history_file)
    if write_header:
        history_writer.writerow(["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"])
        
    num_epochs = 1 if args.smoke_test else EPOCHS
    
    print("\nStarting Training...")
    try:
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, is_image=False
            )
            val_loss, val_acc = evaluate_epoch(
                model, val_loader, criterion, device, is_image=False
            )
            
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | "
                  f"Train: Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val: Loss {val_loss:.4f} Acc {val_acc:.4f} | "
                  f"LR: {current_lr:.1e} | Time: {epoch_time:.1f}s")
                  
            history_writer.writerow([epoch+1, "single", f"{train_loss:.4f}", f"{train_acc:.4f}", 
                                     f"{val_loss:.4f}", f"{val_acc:.4f}", current_lr, f"{epoch_time:.1f}"])
            history_file.flush()
            
            # Save last
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, last_pt_path)
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    history_file.close()
    
    if collected_test_dataset is not None and len(collected_test_dataset) > 0:
        c_loader = DataLoader(collected_test_dataset, batch_size=b_size, shuffle=False)
        c_loss, c_acc = evaluate_epoch(model, c_loader, criterion, device, is_image=False)
        print(f"\nFinal Collected Test Set (unseen signers) - Loss: {c_loss:.4f} Acc: {c_acc:.4f}")
    
    if early_stopping.best_state is not None:
        best_acc = early_stopping.best_loss # Actually we didn't track best_acc directly in ES, but it restored best val_loss.
        # It's cleaner to save the best model out now
        best_pt_path = os.path.join(args.models_dir, "landmark_mlp.pt")
        save_dict = {
            "state_dict": early_stopping.best_state,
            "architecture": "landmark_mlp",
            "class_names": class_names,
            "feature_spec": {
                "input_shape": [2, 21, 3],
                "coordinate_convention": "pixel_space",
                "slot_rule": "sorted_by_wrist_x_asc",
                "pipeline_version": "landmark-v1"
            },
            "mirror_augment": MIRROR_AUGMENT,
            "best_val_loss": early_stopping.best_loss,
            "trained_with_collected": args.include_collected,
            "collected_samples_count": collected_samples_count
        }
        torch.save(save_dict, best_pt_path)
        print(f"\nSaved BEST model to {best_pt_path} (Val Loss: {early_stopping.best_loss:.4f})")
    
if __name__ == "__main__":
    main()
