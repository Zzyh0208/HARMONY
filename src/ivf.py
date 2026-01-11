import torch
from sklearn.cluster import MiniBatchKMeans

class ExpertIVF:
    """
    Inverted File Index for fast retrieval within an Expert.
    """
    def __init__(self, n_centroids=64):
        self.n_centroids = n_centroids
        self.centroids = None
        self.index = [] # List of buckets (tensors)

    def build(self, data_tensor, device):
        # data_tensor: [N, 128]
        if len(data_tensor) < self.n_centroids:
            # Fallback if not enough data
            self.n_centroids = 1
            self.centroids = torch.mean(data_tensor, dim=0, keepdim=True).to(device)
            self.index = [data_tensor]
            return

        kmeans = MiniBatchKMeans(n_clusters=self.n_centroids, batch_size=2048, n_init='auto')
        labels = kmeans.fit_predict(data_tensor.cpu().numpy())
        
        self.centroids = torch.from_numpy(kmeans.cluster_centers_).to(device)
        self.index = []
        for k in range(self.n_centroids):
            self.index.append(data_tensor[labels == k])

    def query(self, query_patches):
        # 1. Coarse Search: Find nearest centroid
        dists = torch.cdist(query_patches, self.centroids)
        nearest = torch.argmin(dists, dim=1)
        
        final_dists = torch.zeros(query_patches.shape[0], device=query_patches.device)
        
        # 2. Fine Search: Look into buckets
        unique_clusters = torch.unique(nearest)
        for k in unique_clusters:
            mask = (nearest == k)
            bucket = self.index[k.item()]
            if len(bucket) > 0:
                # Exact search in bucket
                d = torch.cdist(query_patches[mask], bucket)
                min_d, _ = torch.min(d, dim=1)
                final_dists[mask] = min_d
            else:
                final_dists[mask] = 100.0 # Should not happen
                
        return final_dists