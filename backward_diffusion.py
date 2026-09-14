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
            in_dim = 3
            
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
        """
        device = x.device
        
        h = self.node_mlp(x)
        
        # PyTorch Message Passing (Simple Graph Convolution) over the fragmented topology
        if noisy_edge_index.shape[1] > 0:
            src, dst = noisy_edge_index
            messages = h[src]
            aggr = torch.zeros_like(h)
            aggr.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
            # Add self-loops / residual connection
            h = h + aggr
        
        t_emb = self.time_mlp(t)
        h = h + t_emb[batch_idx]
        
        # Pad graphs for batched Transformer execution
        B = int(batch_idx.max().item() + 1) if len(batch_idx) > 0 else 0
        
        counts = torch.bincount(batch_idx, minlength=B)
        max_nodes = counts.max().item() if len(counts) > 0 else 0
        
        if max_nodes == 0:
            return torch.zeros(candidate_edges.shape[1], device=device)
            
        padded_h = torch.zeros(B, max_nodes, h.size(-1), device=device, dtype=h.dtype)
        padding_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)
        
        # Fast vectorized flat to padded assignment
        cum_counts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(counts, dim=0)])
        seq_i = torch.arange(len(batch_idx), device=device) - cum_counts[batch_idx]
        
        padded_h[batch_idx, seq_i] = h
        padding_mask[batch_idx, seq_i] = False
            
        # Global self-attention to exchange context between fragments
        out_padded = self.transformer(padded_h, src_key_padding_mask=padding_mask)
        
        # Fast vectorized retrieve flat features
        out_flat = out_padded[batch_idx, seq_i]
            
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
    noisy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], device=device)
    batch_idx = torch.tensor([0,0,0,0,0, 1,1,1,1,1], device=device)
    t = torch.tensor([25, 50], device=device)
    candidate_edges = torch.tensor([[0, 2, 5, 8], [1, 4, 6, 9]], device=device)
    
    logits, context = model(x, noisy_edge_index, batch_idx, t, candidate_edges)
    print(f"Predicted Edge Logits Shape: {logits.shape}")
    print(f"Context Embeddings Shape: {context.shape}")
    print(f"Logits output: {logits.detach().cpu().numpy()}")
