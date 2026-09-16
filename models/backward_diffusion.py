import torch
import torch.nn as nn
import math
import os
from utils.utils import load_config
from models.egnn import EGNN

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -embeddings)
        embeddings = t.float().view(-1, 1) * embeddings.view(1, -1)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class BackwardDiffusionModel(nn.Module):
    """
    EGNN + Transformer for Backward Diffusion.
    Takes disjoint neuron fragments, extracts SE(3) invariant features via EGNN over local topology,
    then globally attends to them via a standard Transformer to predict missing edges.
    Fully Scale, Translation, and Rotation Invariant. Memory Efficient!
    """
    def __init__(self, config=None, in_dim=5):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        num_heads = 4
        
        if config is not None:
            bd_params = config.get('backward_diffusion', {})
            hidden_dim = bd_params.get('hidden_dim', hidden_dim)
            num_layers = bd_params.get('num_layers', num_layers)
            num_heads = bd_params.get('num_heads', num_heads)
            
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # We use EGNN over the *local* fragmented edges to extract geometry-aware invariant features
        self.egnn = EGNN(in_node_dim=hidden_dim, hidden_dim=hidden_dim, out_node_dim=hidden_dim, num_layers=num_layers)
        
        # Transformer for global attention between fragmented sub-trees
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Edge prediction head (invariant to SE(3) and Scale since it uses invariant features h and normalized distance)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, noisy_edge_index, batch_idx, t, candidate_edges):
        """
        Args:
            x: (N, 3) coordinates
            noisy_edge_index: (2, E) sparse edges representing the fragmented graph
            batch_idx: (N,) graph assignments
            t: (B,) timesteps
            candidate_edges: (2, E_cand) edge pairs to evaluate
            
        Returns:
            edge_logits: (E_cand,) predictions for each candidate edge
            out_flat: (N, hidden) invariant context embeddings
        """
        device = x.device
        
        # 1. Initialize invariant node features with timestep embeddings
        t_emb = self.time_mlp(t)
        h = t_emb[batch_idx] # (N, hidden_dim)
        
        # 2. Extract invariant features from 3D geometry via EGNN over local topology (memory efficient O(E))
        h, _ = self.egnn(h, x, noisy_edge_index)
        
        # 3. Global Self-Attention via Transformer (PyTorch optimized O(N^2) memory)
        # Remap batch_idx to be contiguous to avoid empty padding rows which cause SDPA NaN gradients
        unique_b, contiguous_batch_idx = torch.unique(batch_idx, return_inverse=True)
        B_eff = len(unique_b)
        counts = torch.bincount(contiguous_batch_idx, minlength=B_eff)
        max_nodes = counts.max().item() if len(counts) > 0 else 0
        
        if max_nodes == 0:
            return torch.zeros(candidate_edges.shape[1], device=device), h
            
        padded_h = torch.zeros(B_eff, max_nodes, h.size(-1), device=device, dtype=h.dtype)
        padding_mask = torch.ones(B_eff, max_nodes, dtype=torch.bool, device=device)
        
        cum_counts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(counts, dim=0)])
        seq_i = torch.arange(len(contiguous_batch_idx), device=device) - cum_counts[contiguous_batch_idx]
        
        padded_h[contiguous_batch_idx, seq_i] = h
        padding_mask[contiguous_batch_idx, seq_i] = False
            
        out_padded = self.transformer(padded_h, src_key_padding_mask=padding_mask)
        out_flat = out_padded[contiguous_batch_idx, seq_i]
        
        # 4. Predict candidate edge logits
        if candidate_edges.size(1) > 0:
            src_nodes = out_flat[candidate_edges[0]]
            dst_nodes = out_flat[candidate_edges[1]]
            
            # Compute graph-level scale for Scale Invariance
            b_counts = counts.unsqueeze(1).float().clamp(min=1)
            mean_x = torch.zeros(B_eff, 3, device=device).scatter_add_(0, contiguous_batch_idx.unsqueeze(1).expand(-1, 3), x) / b_counts
            var_x = torch.zeros(B_eff, 3, device=device).scatter_add_(0, contiguous_batch_idx.unsqueeze(1).expand(-1, 3), (x - mean_x[contiguous_batch_idx])**2) / b_counts
            scale_sq = var_x.sum(dim=-1).clamp(min=1e-6) # (B_eff,)
            
            # Compute invariant relative distance
            src_coords = x[candidate_edges[0]]
            dst_coords = x[candidate_edges[1]]
            sq_dist = torch.sum((src_coords - dst_coords)**2, dim=-1)
            
            # Normalize by graph scale for Scale Invariance
            graph_ids = contiguous_batch_idx[candidate_edges[0]]
            norm_sq_dist = sq_dist / scale_sq[graph_ids]
            log_dist = torch.log1p(norm_sq_dist).unsqueeze(-1)
            
            edge_pairs = torch.cat([src_nodes, dst_nodes, log_dist], dim=-1)
            edge_logits = self.edge_head(edge_pairs).squeeze(-1)
        else:
            edge_logits = torch.empty((0,), device=device)
        
        return edge_logits, out_flat
