import torch
import torch.nn as nn
import collections
import os

def mean_max_pooling(frame_embeddings):
    """
    Helper function for global pooling.
    Args:
        frame_embeddings: [Batch, Time, Dimension]
    Returns:
        [Batch, Dimension]
    """
    assert len(frame_embeddings.shape) == 3 
    (x1, _) = torch.max(frame_embeddings, dim=1)
    x2 = torch.mean(frame_embeddings, dim=1)
    x = x1 + x2
    return x

class AudioNTT2022Encoder(nn.Module):
    """
    Backbone Network (BYOL-A).
    Outputs: [Batch, Time, Dimension]
    """
    def __init__(self, n_mels=64, d=3072, base_d=64, mlp_hidden_d=2048, conv_layers=2, stack=True):
        super().__init__()
        convs = [
            nn.Conv2d(1, base_d, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_d),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        ]
        for c in range(1, conv_layers):
            convs.extend([
                nn.Conv2d(base_d, base_d, 3, stride=1, padding=1),
                nn.BatchNorm2d(base_d),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            ])
        self.features = nn.Sequential(*convs)
        self.conv_d = base_d * (n_mels//(2**conv_layers))
        self.fc = nn.Sequential(
            nn.Linear(self.conv_d, mlp_hidden_d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_hidden_d, d - self.conv_d),
            nn.ReLU(),
        )
        self.stack = stack

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)
        x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) 
        x_fc = self.fc(x)
        x = torch.hstack([x.transpose(1,2), x_fc.transpose(1,2)]).transpose(1,2) if self.stack else x_fc
        return x

class AudioNTT2022(AudioNTT2022Encoder):
    """
    Wrapper for Global Features (used in KNN/SimpleNet baselines).
    Outputs: [Batch, Dimension]
    """
    def __init__(self, n_mels=64, d=3072, mlp_hidden_d=2048):
        super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d)

    def forward(self, x):
        x = super().forward(x)
        x = mean_max_pooling(x)
        return x

class ProjectionHead(nn.Module):
    """
    Learnable Projector for Manifold Disentanglement.
    Maps 3072-dim features to a lower-dimensional (128) embedding space.
    """
    def __init__(self, input_dim=3072, hidden_dim=1024, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# =========================================================================
# [CRITICAL] Robust Weight Loading Function
# =========================================================================
def load_weights(model, weights_path, device):
    """
    Loads pretrained weights into the model, handling key mismatches
    (e.g., removing 'network.' prefix from BYOL-A weights).
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Pretrained weights not found at: {weights_path}")

    print(f"Loading weights from {weights_path}...")
    try:
        # Load state dict
        state_dict = torch.load(weights_path, map_location=device)
        
        # Create a new state dict with mapped keys
        new_state_dict = collections.OrderedDict()
        
        # Get model's current keys to check against
        model_keys = list(model.state_dict().keys())
        
        matched_count = 0
        total_count = len(model_keys)
        
        for k, v in state_dict.items():
            # Fix common prefix mismatches from BYOL-A repo
            new_key = k
            if k.startswith("network."):
                new_key = k.replace("network.", "")
            elif k.startswith("online_network."):
                new_key = k.replace("online_network.", "")
            
            # Check if this key exists in our model
            if new_key in model_keys:
                # Check shape compatibility
                if model.state_dict()[new_key].shape == v.shape:
                    new_state_dict[new_key] = v
                    matched_count += 1
                else:
                    print(f"  ⚠️ Shape mismatch for {new_key}: Expected {model.state_dict()[new_key].shape}, got {v.shape}")
            
        # Load
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        print(f"  ✅ Loaded {matched_count}/{total_count} layers.")
        
        # Safety check: If very few layers matched, something is wrong
        if matched_count < 5:
            print(f"  ❌ WARNING: Very few layers loaded! Check if your weight file is correct.")
            print(f"  Available keys in pth: {list(state_dict.keys())[:5]}...")
            print(f"  Expected keys in model: {model_keys[:5]}...")
            
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise e
    
    model.to(device)
    return model