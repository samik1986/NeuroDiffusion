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

class ValidityHeuristicEvaluator(nn.Module):
    """
    Learned Validity Heuristic H(T1, T2, P, t).
    Evaluates whether a generated path sequence (P) validly connects two disjoint tree fragments (T1 and T2) at diffusion timestep t.
    """
    def __init__(self, config=None, tree_context_dim=128, path_coord_dim=3, path_type_dim=3):
        super().__init__()
        
        hidden_dim = 128
        lstm_layers = 2
        
        if config is not None:
            he_params = config.get('heuristic_evaluator', {})
            hidden_dim = he_params.get('hidden_dim', hidden_dim)
            lstm_layers = he_params.get('lstm_layers', lstm_layers)
            
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
            
        # BiLSTM to encode the generated path sequence P
        seq_input_dim = path_coord_dim + path_type_dim
        self.path_encoder = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim // 2, # Halved for bidirectional
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # MLP to combine T1, T2, Path embeddings, and time embedding
        combined_dim = tree_context_dim * 2 + hidden_dim + hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Outputs a logit for binary classification (valid/invalid)
        )
        
    def forward(self, t1_context, t2_context, path_coords, path_types, t, sequence_lengths=None):
        """
        Args:
            t1_context: (B, tree_context_dim) embedding of fragment 1
            t2_context: (B, tree_context_dim) embedding of fragment 2
            path_coords: (B, max_len, 3) generated 3D coordinates of the path
            path_types: (B, max_len, 3) generated node types (logits or one-hot)
            t: (B,) diffusion timestep
            sequence_lengths: (B,) lengths of the actual paths (if padding is used)
            
        Returns:
            validity_logit: (B, 1) Logit representing probability of validity.
        """
        B = t1_context.size(0)
        
        t_emb = self.time_mlp(t)
        
        # 1. Encode Path P
        path_input = torch.cat([path_coords, path_types.float()], dim=-1)
        
        if sequence_lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                path_input, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hn, cn) = self.path_encoder(packed_input)
            hidden_forward = hn[-2, :, :]
            hidden_backward = hn[-1, :, :]
            path_embedding = torch.cat([hidden_forward, hidden_backward], dim=-1)
        else:
            output, (hn, cn) = self.path_encoder(path_input)
            hidden_forward = hn[-2, :, :]
            hidden_backward = hn[-1, :, :]
            path_embedding = torch.cat([hidden_forward, hidden_backward], dim=-1)
            
        # 2. Combine and Classify
        combined_features = torch.cat([t1_context, t2_context, path_embedding, t_emb], dim=-1)
        
        validity_logit = self.classifier(combined_features)
        return validity_logit
