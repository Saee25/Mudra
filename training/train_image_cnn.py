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
    resolve_classes_and_split, CropImageDataset, train_one_epoch, 
    evaluate_epoch, EarlyStopping
)
from model_defs import MudraMobileNetV2, count_parameters

# --- CONFIG ---
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15
BATCH_SIZE = 64 # Fits on Colab T4
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
PATIENCE = 3

def get_args():
    parser = get_base_parser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()

def set_batchnorm_eval(model):
    """
    Keep all BatchNorm layers in eval mode during fine-tuning.
    Small-batch statistics would ruin the pretrained ImageNet running stats.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def main():
    args = get_args()
    set_seed()
    device = get_device()
    setup_directories(args)
    
    npz_path = os.path.join(args.data_dir, "landmarks", "isl_landmarks.npz")
    crop_dir = os.path.join(args.data_dir, "isl_cropped")
    
    print("Loading split and classes...")
    class_names, train_paths, val_paths = resolve_classes_and_split(
        npz_path, args.outputs_dir, smoke_test=args.smoke_test
    )
    
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    
    print("Initializing datasets...")
    train_dataset = CropImageDataset(crop_dir, train_paths, class_names, is_train=True)
    val_dataset = CropImageDataset(crop_dir, val_paths, class_names, is_train=False)
    
    b_size = 2 if args.smoke_test else BATCH_SIZE
    
    # num_workers > 0 helps image loading, but Windows has issues with multiprocessing in Jupyter
    # Colab handles num_workers=2 nicely.
    num_workers = 2 if not os.name == 'nt' else 0
    if args.smoke_test:
        num_workers = 0
        
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))
                            
    model = MudraMobileNetV2(num_classes).to(device)
    total_params, _ = count_parameters(model)
    
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("Using AMP (Mixed Precision) for training.")
        
    start_epoch = 0
    current_phase = 1
    
    last_pt_path = os.path.join(args.models_dir, "image_cnn_last.pt")
    
    # Initialize optimizer for phase 1 by default
    model.freeze_backbone()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)
    
    if args.resume and os.path.exists(last_pt_path):
        print(f"Resuming from {last_pt_path}")
        checkpoint = torch.load(last_pt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        current_phase = checkpoint["phase"]
        start_epoch = checkpoint["epoch"] + 1
        
        if current_phase == 2:
            model.unfreeze_last_n_blocks(n=4)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2)
            
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed at epoch {start_epoch}, Phase {current_phase}")
        
    history_path = os.path.join(args.outputs_dir, "image_cnn_history.csv")
    write_header = not (args.resume and os.path.exists(history_path))
    
    history_file = open(history_path, mode="a" if not write_header else "w", newline="")
    history_writer = csv.writer(history_file)
    if write_header:
        history_writer.writerow(["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"])
        
    def run_phase(phase_num, epochs_limit, optim, lr_val):
        print(f"\n--- Starting Phase {phase_num} ---")
        _, trainable = count_parameters(model)
        print(f"Params: {total_params:,} Total / {trainable:,} Trainable")
        
        early_stopping = EarlyStopping(patience=PATIENCE)
        
        # If we resumed and are already past the epochs limit for this phase, skip it
        if start_epoch >= epochs_limit and phase_num == 1:
            return
            
        epoch_iter = range(start_epoch if current_phase == phase_num else 0, 1 if args.smoke_test else epochs_limit)
        
        for epoch in epoch_iter:
            start_time = time.time()
            
            if phase_num == 2:
                # model.train() inside train_one_epoch turns BN on, we must turn it back off
                # Hook into train_one_epoch by setting model.train() but overriding BN
                # We do this directly before each batch by modifying train_one_epoch or just doing it here globally
                pass
                
            model.train()
            if phase_num == 2:
                set_batchnorm_eval(model)
                
            # Manual train loop so we can enforce BN eval mode on every batch if needed, 
            # but setting it once after model.train() is sufficient since train_one_epoch 
            # calls model.train() at the start. Wait, train_one_epoch calls model.train(), 
            # so it resets our BN to train mode! We must override it.
            # Instead of changing common.py, we just do a custom loop here.
            
            # --- Custom Train Loop for Model B ---
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                optim.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optim.step()
                    
                total_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                
            train_loss = total_loss / total
            train_acc = correct / total
            # ------------------------------------
            
            val_loss, val_acc = evaluate_epoch(
                model, val_loader, criterion, device, is_image=True
            )
            
            epoch_time = time.time() - start_time
            
            print(f"Phase {phase_num} Epoch {epoch+1:03d} | "
                  f"Train: Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val: Loss {val_loss:.4f} Acc {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
                  
            # Actual epoch index for tracking
            abs_epoch = epoch + (PHASE1_EPOCHS if phase_num == 2 else 0)
            
            history_writer.writerow([abs_epoch+1, f"phase{phase_num}", f"{train_loss:.4f}", f"{train_acc:.4f}", 
                                     f"{val_loss:.4f}", f"{val_acc:.4f}", lr_val, f"{epoch_time:.1f}"])
            history_file.flush()
            
            # Save last
            torch.save({
                "epoch": epoch,
                "phase": phase_num,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
            }, last_pt_path)
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered in Phase {phase_num} at epoch {epoch+1}")
                break
                
        if early_stopping.best_state is not None:
            model.load_state_dict(early_stopping.best_state)
            
    try:
        if current_phase == 1:
            run_phase(1, PHASE1_EPOCHS, optimizer, LR_PHASE1)
            # Switch to phase 2
            current_phase = 2
            model.unfreeze_last_n_blocks(n=4)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2)
            
        if current_phase == 2:
            run_phase(2, PHASE2_EPOCHS, optimizer, LR_PHASE2)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    history_file.close()
    
    # Save best overall
    best_pt_path = os.path.join(args.models_dir, "image_cnn.pt")
    save_dict = {
        "state_dict": model.state_dict(),
        "architecture": "mobilenet_v2_crops",
        "class_names": class_names
    }
    torch.save(save_dict, best_pt_path)
    print(f"\nSaved BEST model to {best_pt_path}")
    
if __name__ == "__main__":
    main()
