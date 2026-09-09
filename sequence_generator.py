import torch
import torch.nn as nn
import math
import os
from utils import load_config

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
    Continuous Diffusion Denoiser for 3D paths.
    Takes noisy path coordinates, time, and tree context, and predicts the added noise.
    """
    def __init__(self, config=None, in_context_dim=128):
        super().__init__()
        
        hidden_dim = 128
        self.max_length = 50
        
        if config is not None:
            sg_params = config.get('sequence_generation', {})
            hidden_dim = sg_params.get('hidden_dim', hidden_dim)
            self.max_length = sg_params.get('max_length', self.max_length)
            
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.context_proj = nn.Linear(in_context_dim, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)
        
        # Transformer to process the sequence of coordinates
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output heads
        self.noise_head = nn.Linear(hidden_dim, 3)
        self.type_head = nn.Linear(hidden_dim, 3)

    def forward(self, noisy_coords, t, context_embedding, padding_mask=None):
        """
        Args:
            noisy_coords: (B, seq_len, 3)
            t: (B,) timesteps
            context_embedding: (B, in_context_dim) graph context to condition on
            padding_mask: (B, seq_len) boolean mask where True means valid, False means padded
            
        Returns:
            noise_pred: (B, seq_len, 3)
            type_logits: (B, seq_len, 3)
        """
        B, seq_len, _ = noisy_coords.shape
        
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, hidden_dim)
        ctx_emb = self.context_proj(context_embedding).unsqueeze(1) # (B, 1, hidden_dim)
        
        x_emb = self.coord_proj(noisy_coords) # (B, seq_len, hidden_dim)
        
        h = x_emb + t_emb + ctx_emb
        
        if padding_mask is not None:
            # PyTorch TransformerEncoder expects True for padded positions to ignore
            key_padding_mask = ~padding_mask
        else:
            key_padding_mask = None
            
        out = self.transformer(h, src_key_padding_mask=key_padding_mask)
        
        noise_pred = self.noise_head(out)
        type_logits = self.type_head(out)
        
        return noise_pred, type_logits
