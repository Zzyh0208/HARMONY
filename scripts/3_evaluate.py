import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config, set_seed, calculate_metrics, get_experiment_id
from src.datasets import DatasetScanner, AudioDataset
from src.models import AudioNTT2022Encoder, ProjectionHead, load_weights
from src.ivf import ExpertIVF 

def evaluate(loader, backbone, proj_head, prototypes, ivf_experts, device):
    backbone.eval(); proj_head.eval()
    all_scores = []; all_labels = []; times = []
    
    prototypes = prototypes.to(device)
    for k in ivf_experts:
        ivf_experts[k].centroids = ivf_experts[k].centroids.to(device)
        ivf_experts[k].index = [b.to(device) for b in ivf_experts[k].index]

    with torch.no_grad():
        for bx, by in tqdm(loader):
            t0 = time.time()
            bx = bx.to(device)
            feat = backbone(bx)
            emb = torch.nn.functional.normalize(proj_head(feat.reshape(-1, feat.shape[-1])), dim=1)
            
            assign = torch.argmin(torch.cdist(emb, prototypes), dim=1)
            scores = torch.zeros(emb.shape[0], device=device)
            
            for eid in torch.unique(assign):
                mask = (assign == eid)
                scores[mask] = ivf_experts[eid.item()].query(emb[mask])
            
            s_scores, _ = torch.max(scores.reshape(bx.shape[0], -1), dim=1)
            times.append((time.time()-t0)/bx.shape[0])
            all_scores.extend(s_scores.cpu().tolist())
            all_labels.extend(by.numpy())
            
    return calculate_metrics(all_labels, all_scores) + (np.mean(times),)

def main():
    config = load_config("conf/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_id = get_experiment_id(config)
    
    backbone = AudioNTT2022Encoder(n_mels=config['audio']['n_mels'], d=config['audio']['feature_d'])
    backbone = load_weights(backbone, config['paths']['pretrained_weights'], device)
    
    proj_head = ProjectionHead(input_dim=config['audio']['feature_d'], output_dim=config['learner']['output_dim'])
    proj_head.load_state_dict(torch.load(os.path.join(config['paths']['output_root'], f"{exp_id}_proj_head.pth"), map_location=device))
    proj_head.to(device)
    
    protos = torch.load(os.path.join(config['paths']['output_root'], f"{exp_id}_expert_prototypes.pt"), weights_only=False)
    prototypes = torch.from_numpy(protos).float() if isinstance(protos, np.ndarray) else protos.float()
    ivf_experts = torch.load(os.path.join(config['paths']['output_root'], f"{exp_id}_ivf_indices.pt"), weights_only=False)
    
    scanner = DatasetScanner(config)
    target = 'all' if exp_id == 'universal' else exp_id
    
    _, _, meta = scanner.scan(target, mode='test')
    machines = set((m['type'], m['id']) for m in meta)
    
    print(f"\n{'Type':<10} {'ID':<5} {'AUC':<8} {'pAUC':<8} {'Time':<10}")
    print("-" * 50)
    
    avgs = []
    for mt, mid in sorted(list(machines)):
        fs, ls, _ = scanner.scan(target, mode='test', machine_type=mt, machine_id=mid)
        if not fs: continue
        
        loader = DataLoader(AudioDataset(fs, ls, config), batch_size=32, shuffle=False, num_workers=4)
        auc, pauc, t = evaluate(loader, backbone, proj_head, prototypes, ivf_experts, device)
        avgs.append(auc)
        print(f"{mt:<10} {mid:<5} {auc:.4f}   {pauc:.4f}   {t*1000:.2f}")
        
    print("-" * 50)
    print(f"Mean AUC: {np.mean(avgs):.4f}")

if __name__ == "__main__":
    main()