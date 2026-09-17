import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import numpy as np

from utils.utils import load_config
from data.data_loader import get_dataloader
from models.forward_diffusion import DiscreteGraphDiffusion
from models.backward_diffusion import BackwardDiffusionModel
from train import plot_scatter_fast, plot_edges_fast

def plot_reconnection_3panel(x, true_edges, noisy_edges, predicted_edges, save_path, epoch, example_idx):
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle(f"Epoch {epoch} - Training Example {example_idx}", fontsize=16)
    x_np = x.cpu().numpy()
    
    # --- Plot 1: Original Clean Graph ---
    ax1 = fig.add_subplot(131, projection='3d')
    plot_scatter_fast(ax1, x_np, c='black', s=5, alpha=0.5)
    if true_edges.shape[1] > 0:
        orig_u, orig_v = true_edges.cpu().numpy()
        plot_edges_fast(ax1, x_np, orig_u, orig_v, c='green', alpha=0.8, linewidth=1.5)
    ax1.set_title("Original Clean Graph")

    # --- Plot 2: Corrupted Graph (Forward Diffusion) ---
    ax2 = fig.add_subplot(132, projection='3d')
    plot_scatter_fast(ax2, x_np, c='black', s=5, alpha=0.5)
    if noisy_edges.shape[1] > 0:
        corr_u, corr_v = noisy_edges.cpu().numpy()
        plot_edges_fast(ax2, x_np, corr_u, corr_v, c='red', linewidth=2.0)
    ax2.set_title("Forward Diffusion (Disconnected Tree)")
    
    # --- Plot 3: Reconnected Graph (Old Inference / Backward Diffusion) ---
    ax3 = fig.add_subplot(133, projection='3d')
    plot_scatter_fast(ax3, x_np, c='black', s=5, alpha=0.5)
    
    # Plot surviving parts (red)
    if noisy_edges.shape[1] > 0:
        corr_u, corr_v = noisy_edges.cpu().numpy()
        plot_edges_fast(ax3, x_np, corr_u, corr_v, c='red', alpha=0.5, linewidth=1.5)
                     
    # Plot generated missing edges (blue)
    if predicted_edges.shape[1] > 0:
        pred_u, pred_v = predicted_edges.cpu().numpy()
        plot_edges_fast(ax3, x_np, pred_u, pred_v, c='blue', linewidth=5.0)
                     
    ax3.set_title("Backward Diffusion (Reconnected)")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main():
    config = load_config('config.yaml')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    out_cfg = config.get('output', {})
    base_dir = out_cfg.get('base_dir', 'output')
    ckpt_dir = os.path.join(base_dir, out_cfg.get('checkpoints_dir', 'checkpoints'))
    
    # Artifact directory for saving
    img_dir = 'output/examples'
    os.makedirs(img_dir, exist_ok=True)
    
    print("Loading training data...")
    train_loader, _ = get_dataloader(config, world_size=1, rank=0)
    
    print("Initializing models...")
    forward_diffusion = DiscreteGraphDiffusion(config=config)
    diff_model = BackwardDiffusionModel(config=config).to(device)
    
    # Get a single batch from the training loader
    batch = next(iter(train_loader))
    while batch is None:
        batch = next(iter(train_loader))
        
    x = batch['x'].to(device)
    true_edge_index = batch['edge_index'].to(device)
    batch_assignment = batch['batch'].to(device)
    
    print("Simulating forward diffusion...")
    t_val = forward_diffusion.num_timesteps - 1
    t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)
    noisy_edges, _ = forward_diffusion.forward_sample(true_edge_index, t_batch)
    
    epoch = 100
    diff_ckpt = os.path.join(ckpt_dir, f"diff_epoch_{epoch}.pth")
    print(f"Loading checkpoint {diff_ckpt}...")
    diff_state = torch.load(diff_ckpt, map_location=device)
    diff_state = {k.replace('module.', ''): v for k, v in diff_state.items()}
    diff_model.load_state_dict(diff_state)
    diff_model.eval()

    # We want 5 examples from the batch
    unique_graphs = torch.unique(batch_assignment)
    examples_to_plot = min(5, len(unique_graphs))
    
    print(f"Found {len(unique_graphs)} unique graphs in batch. Generating {examples_to_plot} examples.")
    
    for i in range(examples_to_plot):
        graph_idx = unique_graphs[i].item()
        mask = (batch_assignment == graph_idx)
        node_indices = torch.where(mask)[0]
        
        if len(node_indices) == 0:
            continue
            
        print(f"Processing Example {i+1} (Graph {graph_idx}) with {len(node_indices)} nodes...")
        x_single = x[mask]
        edge_mask_orig = mask[true_edge_index[0]] & mask[true_edge_index[1]]
        true_edges_single = true_edge_index[:, edge_mask_orig]
        
        edge_mask_corr = mask[noisy_edges[0]] & mask[noisy_edges[1]]
        noisy_edges_single = noisy_edges[:, edge_mask_corr]
        
        mapping = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        mapping[node_indices] = torch.arange(len(node_indices), device=device)
        
        true_edges_single = mapping[true_edges_single]
        noisy_edges_single = mapping[noisy_edges_single]
        
        t_expanded = torch.full((len(x),), t_val, device=device, dtype=torch.long)
        u_nodes = node_indices.cpu().tolist()
        candidates = torch.tensor(list(itertools.combinations(u_nodes, 2)), device=device).T

        if candidates.shape[1] == 0:
             print(f"Example {i+1} has no candidates.")
             continue

        with torch.no_grad():
            logits, _ = diff_model(x, noisy_edges, batch_assignment, t_expanded, candidates)
            
            num_missing_edges = max(1, true_edges_single.shape[1] - noisy_edges_single.shape[1])
            if candidates.shape[1] > num_missing_edges:
                _, topk_indices = torch.topk(logits, num_missing_edges)
                predicted_edges = candidates[:, topk_indices]
            else:
                pred_mask = torch.sigmoid(logits) > 0.5
                predicted_edges = candidates[:, pred_mask]
            
            pred_mask_single = mask[predicted_edges[0]] & mask[predicted_edges[1]]
            predicted_edges_single = predicted_edges[:, pred_mask_single]
            predicted_edges_single = mapping[predicted_edges_single]
            
            save_path = os.path.join(img_dir, f"example_{i+1}_epoch_{epoch}.png")
            plot_reconnection_3panel(x_single, true_edges_single, noisy_edges_single, predicted_edges_single, save_path, epoch, i+1)
            print(f"Generated {save_path}")

if __name__ == '__main__':
    main()
