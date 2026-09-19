import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class LandmarkNormalization(nn.Module):
    """
    In-model normalization for landmark coordinates.
    Input:
        landmarks: float32 [B, 2, 21, 3] in pixel coordinates
        hand_mask: float32 [B, 2] indicating which slots have a hand (1.0 or 0.0)
    Output:
        features: float32 [B, 254]
        
    We use two sets of features:
    1. GLOBAL: centers the present hands around their collective mean (x,y) and 
       scales by the maximum distance to that mean. This preserves the distance 
       and relationship between the two hands (e.g. one hand touching the other).
    2. LOCAL: centers each hand around its OWN wrist (x,y) and scales by its 
       own maximum distance to the wrist. This captures the pure shape of each 
       hand, invariant to where it is relative to the other hand.
       
    This approach ensures invariance to camera resolution, distance, and position,
    using simple tensor ops that export cleanly to ONNX Opset 17.
    """
    def forward(self, landmarks, hand_mask):
        B = landmarks.size(0)
        # m: [B, 2, 1, 1] for broadcasting
        m = hand_mask.unsqueeze(2).unsqueeze(3)
        L = landmarks * m
        
        # --- GLOBAL FEATURES ---
        # 1. Collective mean (center) over all present points.
        # sum over hands (dim=1) and points (dim=2) -> [B, 2] (for x and y)
        sum_xy = L[..., :2].sum(dim=(1, 2))
        # Total present points = 21 * sum(hand_mask)
        # clamp to avoid division by zero if no hands are present (though theoretically we filter those out)
        num_points = (21.0 * hand_mask.sum(dim=1, keepdim=True)).clamp(min=1e-6)
        center_xy = sum_xy / num_points # [B, 2]
        
        # We need a 3D center [B, 3] where Z is 0 (we only translate X and Y)
        zeros_z = torch.zeros_like(center_xy[:, :1])
        center_3d = torch.cat([center_xy, zeros_z], dim=-1) # [B, 3]
        center_3d = center_3d.unsqueeze(1).unsqueeze(2) # [B, 1, 1, 3]
        
        # 2. Global scale: max distance from center in X,Y plane
        # xy_diff: [B, 2, 21, 2]
        xy_diff_global = (L[..., :2] - center_3d[..., :2]) * m[..., :2]
        # norm: [B, 2, 21]
        dist_global = torch.norm(xy_diff_global, dim=-1)
        # max over both hands and points -> [B, 1, 1, 1]
        scale_global = dist_global.flatten(1).max(dim=1)[0].clamp(min=1e-6).view(B, 1, 1, 1)
        
        # 3. Apply global translation and scale
        feat_global = ((L - center_3d) / scale_global) * m
        
        # --- LOCAL FEATURES ---
        # 1. Wrist for each hand (landmark 0)
        # L[:, :, 0:1, :2] gives [B, 2, 1, 2]
        wrist_xy = L[:, :, 0:1, :2]
        wrist_3d = torch.cat([wrist_xy, torch.zeros_like(wrist_xy[..., :1])], dim=-1) # [B, 2, 1, 3]
        
        # 2. Local scale for each hand: max distance from wrist in X,Y plane
        xy_diff_local = (L[..., :2] - wrist_xy) * m[..., :2]
        dist_local = torch.norm(xy_diff_local, dim=-1) # [B, 2, 21]
        # max over points (dim=2) -> [B, 2, 1, 1]
        scale_local = dist_local.max(dim=2, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
        
        # 3. Apply local translation and scale
        feat_local = ((L - wrist_3d) / scale_local) * m
        
        # --- CONCATENATE ---
        feat_global_flat = feat_global.view(B, -1) # 2 * 21 * 3 = 126
        feat_local_flat = feat_local.view(B, -1)   # 126
        
        features = torch.cat([feat_global_flat, feat_local_flat, hand_mask], dim=1) # 126 + 126 + 2 = 254
        return features


class LandmarkMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.norm = LandmarkNormalization()
        # CrossEntropyLoss expects raw logits, so no Softmax here.
        self.mlp = nn.Sequential(
            nn.Linear(254, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, landmarks, hand_mask):
        x = self.norm(landmarks, hand_mask)
        return self.mlp(x)


class InputNormalization(nn.Module):
    """
    Image normalization for MobileNetV2.
    Takes uint8/float32 [0-255] and converts to ImageNet normalized floats.
    Using registered buffers ensures the mean/std are saved with the model 
    and exported into the ONNX graph.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        x = x / 255.0
        return (x - self.mean) / self.std


class MudraMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.norm = InputNormalization()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Replace the original classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.norm(x)
        return self.backbone(x)
        
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
    def unfreeze_last_n_blocks(self, n=4):
        # MobileNetV2 features are a Sequential of blocks (usually 0 to 18)
        num_blocks = len(self.backbone.features)
        start_idx = max(0, num_blocks - n)
        for i in range(start_idx, num_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True


class LandmarkExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, landmarks, hand_mask):
        logits = self.model(landmarks, hand_mask)
        return self.softmax(logits)

class ImageExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, image_nhwc):
        # image_nhwc is [1, 224, 224, 3] 0-255
        x = image_nhwc.permute(0, 3, 1, 2) # [1, 3, 224, 224]
        logits = self.model(x)
        return self.softmax(logits)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run sanity checks")
    args = parser.parse_args()
    
    if args.check:
        print("Running unit checks for LandmarkNormalization...")
        norm = LandmarkNormalization()
        norm.eval()
        
        # Dummy input: 2 batch, 2 hands
        lms = torch.rand(2, 2, 21, 3) * 1000.0 # 0 to 1000 pixel space
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        
        with torch.no_grad():
            feat1 = norm(lms, mask)
            
            # Test 1: Translation invariance
            lms_trans = lms + torch.tensor([500.0, -300.0, 0.0]).view(1, 1, 1, 3)
            feat2 = norm(lms_trans, mask)
            
            # Test 2: Uniform Scale invariance
            lms_scale = lms * 2.0
            feat3 = norm(lms_scale, mask)
            
            # Test 3: Garbage values in absent hand slots
            lms_garbage = lms.clone()
            lms_garbage[0, 1] = torch.rand(21, 3) * 9999.0 # garbage
            lms_garbage[1, 1] = torch.rand(21, 3) * 9999.0 # garbage
            mask_single = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
            
            feat4_clean = norm(lms, mask_single)
            feat4_garbage = norm(lms_garbage, mask_single)
            
            print(f"Translation invariance max diff: {(feat1 - feat2).abs().max().item():.2e}")
            print(f"Uniform scale invariance max diff: {(feat1 - feat3).abs().max().item():.2e}")
            print(f"Garbage slot masking max diff: {(feat4_clean - feat4_garbage).abs().max().item():.2e}")
            
            assert (feat1 - feat2).abs().max().item() < 1e-4
            assert (feat1 - feat3).abs().max().item() < 1e-4
            assert (feat4_clean - feat4_garbage).abs().max().item() < 1e-4
            
        print("LandmarkNormalization checks passed!\n")
        
        print("Testing forward passes...")
        model_a = LandmarkMLP(num_classes=26)
        out_a = model_a(lms, mask)
        print(f"LandmarkMLP output shape: {out_a.shape}")
        
        model_b = MudraMobileNetV2(num_classes=26)
        img = torch.rand(2, 3, 224, 224) * 255.0
        out_b = model_b(img)
        print(f"MudraMobileNetV2 output shape: {out_b.shape}")
        print("Forward pass checks passed!")
