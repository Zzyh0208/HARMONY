import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config, set_seed, get_experiment_id
from src.datasets import DatasetScanner, AudioDataset
from src.models import AudioNTT2022Encoder, ProjectionHead, load_weights
from src.cluster import ClusterManager
from src.ivf import ExpertIVF

def main():
    config = load_config("conf/config.yaml")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_id = get_experiment_id(config)
    proj_path = os.path.join(config['paths']['output_root'], f"{exp_id}_proj_head.pth")
    proto_path = os.path.join(config['paths']['output_root'], f"{exp_id}_expert_prototypes.pt")
    ivf_path = os.path.join(config['paths']['output_root'], f"{exp_id}_ivf_indices.pt")
    
    if os.path.exists(proto_path) and os.path.exists(ivf_path):
        print("Skipping build...")
        return

    scanner = DatasetScanner(config)
    target = 'all' if exp_id == 'universal' else exp_id
    
    files, labels, _ = scanner.scan(dataset_name=target, mode='train')
    
    dataset = AudioDataset(files, labels, config)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    
    backbone = AudioNTT2022Encoder(n_mels=config['audio']['n_mels'], d=config['audio']['feature_d'])
    backbone = load_weights(backbone, config['paths']['pretrained_weights'], device)
    backbone.eval()
    
    proj_head = ProjectionHead(input_dim=config['audio']['feature_d'], output_dim=config['learner']['output_dim'])
    proj_head.load_state_dict(torch.load(proj_path, map_location=device))
    proj_head.to(device).eval()
    
    embeddings = []
    with torch.no_grad():
        for bx, _ in tqdm(loader):
            feat = backbone(bx.to(device))
            flat = feat.reshape(-1, feat.shape[-1])
            emb = torch.nn.functional.normalize(proj_head(flat), dim=1)
            embeddings.append(emb.cpu())
            
    full_emb = torch.cat(embeddings, dim=0)
    
    limit = int(len(full_emb) * 0.8)
    if limit < len(full_emb):
        idx = torch.randperm(len(full_emb))[:limit]
        full_emb = full_emb[idx]
    
    cluster_mgr = ClusterManager(config)
    prototypes, banks = cluster_mgr.build_experts(full_emb)
    
    ivf_experts = {}
    for i, bank in enumerate(banks):
        if len(bank) == 0: continue
        ivf = ExpertIVF(n_centroids=config['cluster']['ivf_centroids'])
        ivf.build(bank.to(device), device)
        ivf.centroids = ivf.centroids.cpu()
        ivf.index = [b.cpu() for b in ivf.index]
        ivf_experts[i] = ivf
        
    torch.save(prototypes, proto_path)
    torch.save(ivf_experts, ivf_path)

if __name__ == "__main__":
    main()