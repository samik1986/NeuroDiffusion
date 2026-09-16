import os
import glob
import torch
import matplotlib
matplotlib.use('Agg')
from utils.utils import load_config
from data.data_loader import get_dataloader
from models.forward_diffusion import DiscreteGraphDiffusion
from models.backward_diffusion import BackwardDiffusionModel
from train import plot_scatter_fast, plot_edges_fast
import matplotlib.pyplot as plt
import itertools

def plot_edges_fast_probs(ax, x_np, u, v, probs, c='blue', linewidth=2.0):
    for i in range(10):
        mask = (probs >= i/10.0) & (probs <= (i+1)/10.0 if i == 9 else probs < (i+1)/10.0)
        if not np.any(mask):
            continue
        u_bin, v_bin = u[mask], v[mask]
        alpha = max(0.05, (i + 1) / 10.0)
        plot_edges_fast(ax, x_np, u_bin, v_bin, c=c, alpha=alpha, linewidth=linewidth)

def plot_edges_fast_probs(ax, x_np, u, v, probs, c='blue', linewidth=2.0):
    for i in range(10):
        mask = (probs >= i/10.0) & (probs <= (i+1)/10.0 if i == 9 else probs < (i+1)/10.0)
        if not np.any(mask):
            continue
        u_bin, v_bin = u[mask], v[mask]
        alpha = max(0.05, (i + 1) / 10.0)
        plot_edges_fast(ax, x_np, u_bin, v_bin, c=c, alpha=alpha, linewidth=linewidth)

def plot_edges_fast_probs(ax, x_np, u, v, probs, c='blue', linewidth=2.0):
    for i in range(10):
        mask = (probs >= i/10.0) & (probs <= (i+1)/10.0 if i == 9 else probs < (i+1)/10.0)
        if not np.any(mask):
            continue
        u_bin, v_bin = u[mask], v[mask]
        alpha = max(0.05, (i + 1) / 10.0)
        plot_edges_fast(ax, x_np, u_bin, v_bin, c=c, alpha=alpha, linewidth=linewidth)

def plot_reconnection(x, true_edges, noisy_edges, predicted_edges, save_path, epoch):
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle(f"Epoch {epoch} Reconnection Process", fontsize=16)
    x_np = x.cpu().numpy()
    
    # --- Plot 1: Original Graph ---
    ax1 = fig.add_subplot(141, projection='3d')
    plot_scatter_fast(ax1, x_np, c='black', s=5, alpha=0.5)
    if true_edges.shape[1] > 0:
        orig_u, orig_v = true_edges.cpu().numpy()
        plot_edges_fast(ax1, x_np, orig_u, orig_v, c='green', alpha=0.8, linewidth=1.5)
    ax1.set_title("1. Original Clean Graph")
    
    # --- Plot 2: Corrupted Graph (Forward Diffusion) ---
    ax2 = fig.add_subplot(142, projection='3d')
    plot_scatter_fast(ax2, x_np, c='black', s=5, alpha=0.5)
    if true_edges.shape[1] > 0:
        orig_u, orig_v = true_edges.cpu().numpy()
        plot_edges_fast(ax2, x_np, orig_u, orig_v, c='lightgrey', alpha=0.2, linewidth=1.0)
    if noisy_edges.shape[1] > 0:
        corr_u, corr_v = noisy_edges.cpu().numpy()
        plot_edges_fast(ax2, x_np, corr_u, corr_v, c='red', linewidth=2.0)
    ax2.set_title("2. Disconnected Subgraphs (After Forward Diffusion)")
    
    # --- Plot 3: Reconnected Graph (Backward Diffusion) ---
    ax3 = fig.add_subplot(143, projection='3d')
    plot_scatter_fast(ax3, x_np, c='black', s=5, alpha=0.5)
    
    # Plot surviving parts
    if noisy_edges.shape[1] > 0:
        corr_u, corr_v = noisy_edges.cpu().numpy()
        plot_edges_fast(ax3, x_np, corr_u, corr_v, c='red', alpha=0.5, linewidth=1.5)
                     
    # Plot generated missing edges
    if predicted_edges.shape[1] > 0:
        pred_u, pred_v = predicted_edges.cpu().numpy()
        print(f"Plotting {len(pred_u)} blue edges! First 5 u: {pred_u[:5]}, v: {pred_v[:5]}")
        if pred_probs is not None:
            plot_edges_fast_probs(ax3, x_np, pred_u, pred_v, pred_probs.cpu().float().numpy(), c='blue', linewidth=2.0)
        else:
            plot_edges_fast(ax3, x_np, pred_u, pred_v, c='blue', linewidth=5.0)
                     
    ax3.set_title("3. Model Reconnecting Subgraphs (Backward Diffusion)")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main():
    config = load_config('config.yaml')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    out_cfg = config.get('output', {})
    base_dir = out_cfg.get('base_dir', 'output')
    ckpt_dir = os.path.join(base_dir, out_cfg.get('checkpoints_dir', 'checkpoints'))
    img_dir = os.path.join(base_dir, out_cfg.get('images_dir', 'corrupted_disjoint_trees'))
    os.makedirs(img_dir, exist_ok=True)
    
    _, val_loader = get_dataloader(config, world_size=1, rank=0)
    
    forward_diffusion = DiscreteGraphDiffusion(config=config)
    diff_model = BackwardDiffusionModel(config=config).to(device)
    
    # Get a single batch of data to use across all epochs
    batch = next(iter(val_loader))
    while batch is None:
        batch = next(iter(val_loader))
        
    x = batch['x'].to(device)
    true_edge_index = batch['edge_index'].to(device)
    batch_assignment = batch['batch'].to(device)
    
    # Pre-compute the corruption and candidates so it's identical across all epochs
    t_val = forward_diffusion.num_timesteps - 1
    t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)
    noisy_edges, _ = forward_diffusion.forward_sample(true_edge_index, t_batch)
    
    mask = (batch_assignment == 0)
    node_indices = torch.where(mask)[0]
    
    if len(node_indices) == 0:
        print("First graph has no nodes. Aborting.")
        return
        
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

    # Find all checkpoints
    epochs = [3]
    
    if not epochs:
        print("No checkpoints found!")
        return
        
    print(f"Found {len(epochs)} epochs to process.")

    for epoch in epochs:
        diff_ckpt = os.path.join(ckpt_dir, f"diff_epoch_{epoch}.pth")
        diff_state = torch.load(diff_ckpt, map_location=device)
        diff_state = {k.replace('module.', ''): v for k, v in diff_state.items()}
        diff_model.load_state_dict(diff_state)
        diff_model.eval()
        
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
            print(f"Num true edges: {true_edges_single.shape[1]}, Num noisy edges: {noisy_edges_single.shape[1]}")
            print(f"Candidates shape: {candidates.shape}, Predicted edges shape: {predicted_edges.shape}, Mapped single shape: {predicted_edges_single.shape}")
            
            save_path = os.path.join(img_dir, f"epoch_{epoch}_backward_reconnection.png")
            plot_reconnection(x_single, true_edges_single, noisy_edges_single, predicted_edges_single, save_path, epoch)
            print(f"Generated {save_path}")

if __name__ == '__main__':
    main()
