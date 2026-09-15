import torch
import torch.nn as nn
import math
import os
from utils.utils import load_config

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
        
        # Takes concatenated t1 and t2 contexts
        self.context_proj = nn.Linear(in_context_dim * 2, hidden_dim)
        
        # Empty state embedding for when t2 is missing (branch extension)
        self.t2_empty_state = nn.Parameter(torch.zeros(in_context_dim))
        
        self.coord_proj = nn.Linear(3, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.noise_head = nn.Linear(hidden_dim, 3)
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
            noise_pred: (B, seq_len, 3)
            type_logits: (B, seq_len, 3)
        """
        B, seq_len, _ = noisy_coords.shape
        
        if t2_context is None:
            t2_context = self.t2_empty_state.unsqueeze(0).expand(B, -1)
            
        combined_context = torch.cat([t1_context, t2_context], dim=-1)
        
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, hidden_dim)
        ctx_emb = self.context_proj(combined_context).unsqueeze(1) # (B, 1, hidden_dim)
        
        x_emb = self.coord_proj(noisy_coords) # (B, seq_len, hidden_dim)
        
        pos = torch.arange(seq_len, device=noisy_coords.device).unsqueeze(0)
        p_emb = self.pos_emb(pos) # (1, seq_len, hidden_dim)
        
        h = x_emb + t_emb + ctx_emb + p_emb
        
        if padding_mask is not None:
            # PyTorch TransformerEncoder expects True for padded positions to ignore
            key_padding_mask = ~padding_mask
        else:
            key_padding_mask = None
            
        out = self.transformer(h, src_key_padding_mask=key_padding_mask)
        
        noise_pred = self.noise_head(out)
        type_logits = self.type_head(out)
        
        return noise_pred, type_logits
        
    @torch.no_grad()
    def sample(self, t1_context, t2_context, num_timesteps=100, beta_start=1e-4, beta_end=0.02):
        """
        Reverse DDPM loop to generate a sequence from pure noise.
        Saves intermediate states for visualization.
        
        Returns:
            final_coords: (B, seq_len, 3)
            final_types: (B, seq_len)
            history: list of dicts with intermediate coordinates
        """
        device = t1_context.device
        B = t1_context.size(0)
        
        # Setup noise schedule (linear)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Start from pure Gaussian noise
        x_t = torch.randn(B, self.max_length, 3, device=device)
        
        history = []
        
        for t_step in reversed(range(num_timesteps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred, type_logits = self.forward(x_t, t, t1_context, t2_context)
            
            # Record intermediate state for 5 evenly spaced intervals to match the 5 subplots
            if t_step in [num_timesteps - 1, (num_timesteps * 3 // 4) - 1, (num_timesteps * 2 // 4) - 1, (num_timesteps // 4) - 1, 0]:
                history.append({
                    't': t_step,
                    'coords': x_t.clone()
                })
            
            if t_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
                
            # DDPM reverse step
            alpha_t = alphas[t_step]
            alpha_bar_t = alphas_cumprod[t_step]
            
            # Posterior mean
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
            # Add variance
            x_t = x_t + torch.sqrt(betas[t_step]) * noise
            
        final_types = type_logits.argmax(dim=-1)
        return x_t, final_types, history
