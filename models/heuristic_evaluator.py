import torch
import torch.nn as nn
import math
import os
from utils.utils import load_config
from models.egnn import EGNN

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ValidityHeuristicEvaluator(nn.Module):
    """
    Learned Validity Heuristic H(T1, T2, P, t) using EGNN.
    Evaluates whether a generated path sequence (P) validly connects two disjoint tree fragments (T1 and T2) at diffusion timestep t.
    Fully SE(3) Invariant.
    """
    def __init__(self, config=None, tree_context_dim=128, path_coord_dim=3, path_type_dim=3):
        super().__init__()
        
        hidden_dim = 128
        num_layers = 2
        
        if config is not None:
            he_params = config.get('heuristic_evaluator', {})
            hidden_dim = he_params.get('hidden_dim', hidden_dim)
            num_layers = he_params.get('egnn_layers', num_layers)
            
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
            
        # We replace the BiLSTM with an EGNN
        # Initial node features will just be the projected path types
        self.type_proj = nn.Linear(path_type_dim, hidden_dim)
        
        self.egnn = EGNN(in_node_dim=hidden_dim, hidden_dim=hidden_dim, out_node_dim=hidden_dim, num_layers=num_layers)
        
        # MLP to combine T1, T2, Path embeddings, and time embedding
        combined_dim = tree_context_dim * 2 + hidden_dim + hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Outputs a scalar predicting the MSE structural gap
        )
        
    def forward(self, t1_context, t2_context, path_coords, path_types, t, sequence_lengths=None):
        """
        Args:
            t1_context: (B, tree_context_dim) embedding of fragment 1
            t2_context: (B, tree_context_dim) embedding of fragment 2
            path_coords: (B, seq_len, 3) generated 3D coordinates of the path
            path_types: (B, seq_len, 3) generated node types (logits or one-hot)
            t: (B,) diffusion timestep
            sequence_lengths: (B,) lengths of the actual paths (if padding is used)
            
        Returns:
            validity_logit: (B, 1) Logit representing probability of validity.
        """
        device = path_coords.device
        B, seq_len, _ = path_coords.shape
        
        t_emb = self.time_mlp(t)
        
        # 1. Encode Path P using EGNN
        # Initial invariant features h are the projected types
        h = self.type_proj(path_types.float()) # (B, seq_len, hidden_dim)
        
        # Construct 1D chain edges for the batch
        row_list, col_list = [], []
        for i in range(seq_len - 1):
            row_list.extend([i, i+1])
            col_list.extend([i+1, i])
            
        base_row = torch.tensor(row_list, dtype=torch.long, device=device)
        base_col = torch.tensor(col_list, dtype=torch.long, device=device)
        
        edge_indices = []
        for b in range(B):
            offset = b * seq_len
            edge_indices.append(torch.stack([base_row + offset, base_col + offset], dim=0))
            
        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
        h_flat = h.view(B * seq_len, -1)
        x_flat = path_coords.view(B * seq_len, 3)
        
        # Forward pass through EGNN
        h_out, _ = self.egnn(h_flat, x_flat, edge_index)
        
        h_out = h_out.view(B, seq_len, -1)
        
        # Pool the invariant features across the sequence
        if sequence_lengths is not None:
            # Masked mean pooling
            mask = torch.arange(seq_len, device=device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
            mask = mask.float().unsqueeze(-1)
            path_embedding = (h_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            path_embedding = h_out.mean(dim=1)
            
        # 2. Combine and Predict Gap
        combined_features = torch.cat([t1_context, t2_context, path_embedding, t_emb], dim=-1)
        
        predicted_gap = self.classifier(combined_features)
        return predicted_gap
