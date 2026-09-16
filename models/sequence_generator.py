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

class SequenceGenerator(nn.Module):
    """
    Continuous Diffusion Denoiser for 3D paths using a 1D-Chain EGNN.
    Takes noisy path coordinates, time, and tree context, and equivariantly predicts the added noise.
    """
    def __init__(self, config=None, in_context_dim=128):
        super().__init__()
        
        hidden_dim = 128
        self.max_length = 50
        num_layers = 4
        
        if config is not None:
            sg_params = config.get('sequence_generation', {})
            hidden_dim = sg_params.get('hidden_dim', hidden_dim)
            self.max_length = sg_params.get('max_length', self.max_length)
            
        self.dim = hidden_dim
            
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Takes concatenated t1 and t2 contexts
        self.context_proj = nn.Linear(in_context_dim * 2, hidden_dim)
        
        # Empty state embedding for when t2 is missing (branch extension)
        self.t2_empty_state = nn.Parameter(torch.zeros(in_context_dim))
        
        # We replace the Transformer with a 1D-Chain EGNN
        self.egnn = EGNN(in_node_dim=hidden_dim, hidden_dim=hidden_dim, out_node_dim=hidden_dim, num_layers=num_layers)
        
        # Invariant node types
        self.type_head = nn.Linear(hidden_dim, 3)
        
        # Positional embeddings for the sequence
        self.pos_emb = nn.Embedding(1000, hidden_dim)

    def forward(self, noisy_coords, t, t1_context, t2_context=None, padding_mask=None):
        """
        Args:
            noisy_coords: (B, seq_len, 3)
            t: (B,) timesteps
            t1_context: (B, in_context_dim) fragment 1 context
            t2_context: (B, in_context_dim) fragment 2 context, or None for branch extension
            padding_mask: (B, seq_len) boolean mask where True means valid, False means padded
            
        Returns:
            noise_pred: (B, seq_len, 3) predicted translation-invariant, rotation-equivariant noise
            type_logits: (B, seq_len, 3) invariant node type predictions
        """
        device = noisy_coords.device
        B, seq_len, _ = noisy_coords.shape
        
        if t2_context is None:
            t2_context = self.t2_empty_state.unsqueeze(0).expand(B, -1)
            
        combined_context = torch.cat([t1_context, t2_context], dim=-1)
        
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, hidden_dim)
        ctx_emb = self.context_proj(combined_context).unsqueeze(1) # (B, 1, hidden_dim)
        
        # Parameter-free Sinusoidal Positional Embeddings
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = pos[:, None] * embeddings[None, :]
        p_emb = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            p_emb = torch.nn.functional.pad(p_emb, (0, 1))
        p_emb = p_emb.unsqueeze(0) # (1, seq_len, hidden_dim)
        
        # Initial invariant features for the EGNN
        # We DO NOT project coordinates into features anymore. The coordinates are passed separately to EGNN.
        h = t_emb + ctx_emb + p_emb # (B, seq_len, hidden_dim)
        
        # Construct 1D chain edges for the batch
        # For each graph in the batch, connect i to i+1 and i+1 to i
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
            
        # Flatten batch for EGNN
        h_flat = h.view(B * seq_len, -1)
        x_flat = noisy_coords.view(B * seq_len, 3)
        
        # Forward pass through EGNN
        h_out, x_out = self.egnn(h_flat, x_flat, edge_index)
        
        # Reshape back to batch format
        h_out = h_out.view(B, seq_len, -1)
        x_out = x_out.view(B, seq_len, 3)
        
        # The EGNN output coordinates (x_out) are x_flat + delta_x.
        # delta_x is our predicted equivariant noise.
        noise_pred = x_out - noisy_coords
        
        # Predict node types from invariant features
        type_logits = self.type_head(h_out)
        
        return noise_pred, type_logits
        
    @torch.no_grad()
    def sample(self, t1_context, t2_context, num_timesteps=100, beta_start=1e-4, beta_end=0.02):
        device = t1_context.device
        B = t1_context.size(0)
        
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        x_t = torch.randn(B, self.max_length, 3, device=device)
        history = []
        
        for t_step in reversed(range(num_timesteps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            
            noise_pred, type_logits = self.forward(x_t, t, t1_context, t2_context)
            
            if t_step in [num_timesteps - 1, (num_timesteps * 3 // 4) - 1, (num_timesteps * 2 // 4) - 1, (num_timesteps // 4) - 1, 0]:
                history.append({
                    't': t_step,
                    'coords': x_t.clone()
                })
            
            if t_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
                
            alpha_t = alphas[t_step]
            alpha_bar_t = alphas_cumprod[t_step]
            
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            x_t = x_t + torch.sqrt(betas[t_step]) * noise
            
        final_types = type_logits.argmax(dim=-1)
        return x_t, final_types, history
