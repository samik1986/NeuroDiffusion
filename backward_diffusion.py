import torch
import torch.nn as nn
import math
import os
from utils import load_config

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
    Graph Transformer for Backward Diffusion.
    Takes disjoint neuron fragments and globally attends to them to predict missing edges.
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
            laplacian_k = bd_params.get('laplacian_k', 8)
            in_dim = 5 + laplacian_k
            
        self.node_mlp = nn.Linear(in_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer for global attention between fragmented sub-trees
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Edge prediction head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, inv_features, batch_idx, t, candidate_edges):
        """
        Args:
            x: (N, 3) coordinates
            inv_features: (N, 2) invariant features
            batch_idx: (N,) graph assignments
            t: (B,) timesteps
            candidate_edges: (2, E_cand) edge pairs to evaluate
            
        Returns:
            edge_logits: (E_cand,) predictions for each candidate edge
        """
        device = x.device
        
        node_feats = torch.cat([x, inv_features], dim=-1)
        h = self.node_mlp(node_feats)
        
        t_emb = self.time_mlp(t)
        h = h + t_emb[batch_idx]
        
        # Pad graphs for batched Transformer execution
        unique_batches = torch.unique(batch_idx)
        B = len(unique_batches)
        
        node_counts = [(batch_idx == i).sum().item() for i in range(B)]
        max_nodes = max(node_counts) if node_counts else 0
        
        if max_nodes == 0:
            return torch.zeros(candidate_edges.shape[1], device=device)
            
        padded_h = torch.zeros(B, max_nodes, h.size(-1), device=device)
        padding_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)
        
        # Manual flat to padded assignment
        flat_to_padded_idx = {}
        curr_idx = 0
        for i in range(B):
            n_nodes = node_counts[i]
            padded_h[i, :n_nodes] = h[curr_idx:curr_idx+n_nodes]
            padding_mask[i, :n_nodes] = False
            for j in range(n_nodes):
                flat_to_padded_idx[curr_idx + j] = (i, j)
            curr_idx += n_nodes
            
        # Global self-attention to exchange context between fragments
        out_padded = self.transformer(padded_h, src_key_padding_mask=padding_mask)
        
        # Retrieve flat features
        out_flat = torch.zeros_like(h)
        for flat_i, (b_i, seq_i) in flat_to_padded_idx.items():
            out_flat[flat_i] = out_padded[b_i, seq_i]
            
        # Predict candidate edge logits
        src_nodes = out_flat[candidate_edges[0]]
        dst_nodes = out_flat[candidate_edges[1]]
        
        edge_pairs = torch.cat([src_nodes, dst_nodes], dim=-1)
        edge_logits = self.edge_head(edge_pairs).squeeze(-1)
        
        return edge_logits, out_flat

if __name__ == '__main__':
    print("Testing Backward Diffusion Graph Transformer...")
    
    config = None
    if os.path.exists('config.yaml'):
        config = load_config('config.yaml')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for backward diffusion operations")
    
    model = BackwardDiffusionModel(config=config).to(device)
    
    # Mock inputs
    N = 10
    B = 2
    x = torch.rand(N, 3, device=device)
    inv_features = torch.rand(N, 2 + model.node_mlp.in_features - 5, device=device)
    batch_idx = torch.tensor([0,0,0,0,0, 1,1,1,1,1], device=device)
    t = torch.tensor([25, 50], device=device)
    candidate_edges = torch.tensor([[0, 2, 5, 8], [1, 4, 6, 9]], device=device)
    
    logits, context = model(x, inv_features, batch_idx, t, candidate_edges)
    print(f"Predicted Edge Logits Shape: {logits.shape}")
    print(f"Context Embeddings Shape: {context.shape}")
    print(f"Logits output: {logits.detach().cpu().numpy()}")
