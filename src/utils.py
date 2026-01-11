import torch
import numpy as np
import random
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def visualize_tsne(features, labels, title, save_path, n_samples=5000):
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    print(f"Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, palette='tab10', s=15, alpha=0.7)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE saved to {save_path}")

from sklearn.metrics import roc_auc_score

def calculate_metrics(labels, scores):
    auc = roc_auc_score(labels, scores)
    

    try:
        p_auc = roc_auc_score(labels, scores, max_fpr=0.1)
    except:
        p_auc = 0.5
        
    return auc, p_auc

def get_experiment_id(config):
    use_mimii = config['dataset']['use_mimii']
    use_dcase = config['dataset']['use_dcase']
    
    if use_mimii and use_dcase:
        return "universal"
    elif use_mimii:
        return "mimii"
    elif use_dcase:
        return "dcase"
    else:
        return "unknown"