import torch
import torch.nn as nn
import os
from utils import load_config

class ValidityHeuristicEvaluator(nn.Module):
    """
    Learned Validity Heuristic H(T1, T2, P).
    Evaluates whether a generated path sequence (P) validly connects two disjoint tree fragments (T1 and T2).
    """
    def __init__(self, config=None, tree_context_dim=128, path_coord_dim=3, path_type_dim=3):
        super().__init__()
        
        hidden_dim = 128
        lstm_layers = 2
        
        if config is not None:
            he_params = config.get('heuristic_evaluator', {})
            hidden_dim = he_params.get('hidden_dim', hidden_dim)
            lstm_layers = he_params.get('lstm_layers', lstm_layers)
            
        # BiLSTM to encode the generated path sequence P
        # Input to sequence: coordinates (3) + node types (3) = 6
        seq_input_dim = path_coord_dim + path_type_dim
        self.path_encoder = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim // 2, # Halved for bidirectional
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # MLP to combine T1, T2, and Path embeddings
        # Input: T1_emb (tree_context_dim) + T2_emb (tree_context_dim) + Path_emb (hidden_dim)
        combined_dim = tree_context_dim * 2 + hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Outputs a logit for binary classification (valid/invalid)
        )
        
    def forward(self, t1_context, t2_context, path_coords, path_types, sequence_lengths=None):
        """
        Args:
            t1_context: (B, tree_context_dim) embedding of fragment 1
            t2_context: (B, tree_context_dim) embedding of fragment 2
            path_coords: (B, max_len, 3) generated 3D coordinates of the path
            path_types: (B, max_len, 3) generated node types (logits or one-hot)
            sequence_lengths: (B,) lengths of the actual paths (if padding is used)
            
        Returns:
            validity_logit: (B, 1) Logit representing probability of validity.
        """
        B = t1_context.size(0)
        
        # 1. Encode Path P
        path_input = torch.cat([path_coords, path_types], dim=-1)
        
        if sequence_lengths is not None:
            # Pack sequence for dynamic lengths
            packed_input = nn.utils.rnn.pack_padded_sequence(
                path_input, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hn, cn) = self.path_encoder(packed_input)
            
            # Extract final hidden state from both directions
            # hn is of shape (num_layers * num_directions, batch, hidden_size)
            hidden_forward = hn[-2, :, :]
            hidden_backward = hn[-1, :, :]
            path_embedding = torch.cat([hidden_forward, hidden_backward], dim=-1)
        else:
            # Assume all sequences are equal length (or fully padded to max_len but without pack_padded)
            # Usually fallback for simple testing without lengths
            output, (hn, cn) = self.path_encoder(path_input)
            hidden_forward = hn[-2, :, :]
            hidden_backward = hn[-1, :, :]
            path_embedding = torch.cat([hidden_forward, hidden_backward], dim=-1)
            
        # 2. Combine and Classify
        combined_features = torch.cat([t1_context, t2_context, path_embedding], dim=-1)
        
        validity_logit = self.classifier(combined_features)
        return validity_logit

if __name__ == '__main__':
    print("Testing Validity Heuristic Evaluator...")
    
    config = None
    if os.path.exists('config.yaml'):
        config = load_config('config.yaml')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for testing")
    
    evaluator = ValidityHeuristicEvaluator(config=config).to(device)
    
    # Mock inputs
    B = 8
    max_len = 15
    t1_context = torch.rand(B, 128, device=device)
    t2_context = torch.rand(B, 128, device=device)
    
    path_coords = torch.rand(B, max_len, 3, device=device)
    path_types = torch.rand(B, max_len, 3, device=device)
    
    # Mock varying sequence lengths for the paths
    seq_lengths = torch.randint(3, max_len + 1, (B,), device=device)
    
    logits = evaluator(t1_context, t2_context, path_coords, path_types, sequence_lengths=seq_lengths)
    probabilities = torch.sigmoid(logits)
    
    print(f"Valid Logits Output Shape: {logits.shape}")
    print(f"Sample Probabilities (Valid vs Invalid):\n{probabilities.detach().cpu().numpy().flatten()}")
