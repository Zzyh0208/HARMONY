import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config, set_seed, ensure_dir, get_experiment_id
from src.datasets import DatasetScanner, AudioDataset
from src.models import AudioNTT2022Encoder, ProjectionHead, load_weights
from src.cluster import ClusterManager

def main():
    config = load_config("conf/config.yaml")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(config['paths']['output_root'])
    
    exp_id = get_experiment_id(config)
    save_path = os.path.join(config['paths']['output_root'], f"{exp_id}_proj_head.pth")
    
    if os.path.exists(save_path):
        print(f"Skipping {exp_id} training...")
        return

    scanner = DatasetScanner(config)
    target = 'all' if exp_id == 'universal' else exp_id
    
    files, labels, _ = scanner.scan(dataset_name=target, mode='train')
    
    dataset = AudioDataset(files, labels, config)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
    
    backbone = AudioNTT2022Encoder(n_mels=config['audio']['n_mels'], d=config['audio']['feature_d'])
    backbone = load_weights(backbone, config['paths']['pretrained_weights'], device)
    backbone.eval()
    
    cluster_mgr = ClusterManager(config)
    cluster_mgr.fit_pca(loader, backbone, device)
    feats, labels = cluster_mgr.generate_pseudo_labels(loader, backbone, device)
    
    proj_head = ProjectionHead(
        input_dim=config['audio']['feature_d'],
        output_dim=config['learner']['output_dim']
    ).to(device)
    
    optimizer = optim.Adam(proj_head.parameters(), lr=config['learner']['lr'])
    criterion = nn.CrossEntropyLoss()
    classifier = nn.Linear(config['learner']['output_dim'], config['learner']['n_anchors']).to(device)
    
    train_loader = DataLoader(TensorDataset(feats, labels), batch_size=config['learner']['batch_size'], shuffle=True)
    
    proj_head.train()
    for epoch in range(config['learner']['epochs']):
        total = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            emb = torch.nn.functional.normalize(proj_head(bx), dim=1)
            loss = criterion(classifier(emb), by)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1} Loss: {total/len(train_loader):.4f}")
        
    torch.save(proj_head.state_dict(), save_path)

if __name__ == "__main__":
    main()