import numpy as np
import torch
import os
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

class ClusterManager:
    def __init__(self, config):
        self.config = config
        self.pca = None
        self.kmeans = None
        self.pca_anchors = None
        
    def fit_pca(self, data_loader, model, device, max_samples=50000):
        """
        Phase 1 Cold Start: Extract features and fit PCA.
        """
        print("--- [ClusterManager] Fitting PCA (Cold Start) ---")
        model.eval()
        features = []
        
        # Collect a subset of features
        with torch.no_grad():
            for batch_x, _ in tqdm(data_loader, desc="Extracting for PCA"):
                # batch_x: [B, 1, F, T]
                feat = model(batch_x.to(device)) # [B, T, D]
                # Flatten patches
                feat = feat.reshape(-1, feat.shape[-1]).cpu().numpy()
                
                features.append(feat)
                if len(features) * feat.shape[0] > max_samples:
                    break
                    
        X = np.concatenate(features, axis=0)
        if X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X = X[indices]
            
        print(f"Fitting PCA on {X.shape}...")
        self.pca = PCA(n_components=self.config['learner']['pca_dim'], random_state=42)
        X_pca = self.pca.fit_transform(X)
        print(f"PCA Explained Variance: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Find Anchors in PCA space
        print("Mining Anchors...")
        kmeans = MiniBatchKMeans(n_clusters=32, batch_size=4096, n_init='auto', random_state=42)
        kmeans.fit(X_pca)
        
        # Select compact anchors (simplified logic)
        # In a real run, you'd calculate compactness. Here we pick centroids directly for speed.
        # Taking the top N_ANCHORS centroids
        self.pca_anchors = kmeans.cluster_centers_[:self.config['learner']['n_anchors']]
        return self.pca, self.pca_anchors

    def generate_pseudo_labels(self, data_loader, model, device):
        """
        Generate labels for Phase 1 training.
        """
        print("--- [ClusterManager] Generating Pseudo-Labels ---")
        model.eval()
        all_labels = []
        all_features = [] # Optional: return features to save cache
        
        anchors_tensor = torch.tensor(self.pca_anchors, dtype=torch.float32).to(device)
        pca_comp = torch.tensor(self.pca.components_.T, dtype=torch.float32).to(device)
        pca_mean = torch.tensor(self.pca.mean_, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            for batch_x, _ in tqdm(data_loader, desc="Labeling"):
                feat = model(batch_x.to(device)) # [B, T, D]
                B, T, D = feat.shape
                feat_flat = feat.reshape(-1, D)
                
                # Project to PCA on GPU
                feat_pca = torch.matmul(feat_flat - pca_mean, pca_comp)
                
                # Find nearest anchor
                dists = torch.cdist(feat_pca, anchors_tensor)
                labels = torch.argmin(dists, dim=1)
                
                all_labels.append(labels.cpu())
                all_features.append(feat_flat.cpu()) # Keep features for training MLP directly
                
        return torch.cat(all_features), torch.cat(all_labels)

    def build_experts(self, features_128_tensor):
        """
        Phase 2: Cluster the *learned* 128-dim features to find Experts.
        """
        print(f"--- [ClusterManager] Building Experts on {features_128_tensor.shape} ---")
        n_experts = self.config['cluster']['n_experts']
        
        kmeans = MiniBatchKMeans(n_clusters=n_experts, batch_size=10000, n_init='auto', random_state=42)
        X_np = features_128_tensor.cpu().numpy()
        
        labels = kmeans.fit_predict(X_np)
        prototypes = kmeans.cluster_centers_
        
        # Separate data into Memory Banks
        memory_banks = []
        for i in range(n_experts):
            mask = (labels == i)
            bank = features_128_tensor[mask]
            memory_banks.append(bank)
            print(f"  Expert {i}: {len(bank)} patches")
            
        return prototypes, memory_banks