import torch
import torch.nn as nn
import os
from utils import load_config

class SequenceGenerator(nn.Module):
    """
    Autoregressive model to generate continuous 3D coordinates extending from a fragment endpoint.
    Also predicts whether the generated node should branch internally.
    """
    def __init__(self, config=None, in_context_dim=128):
        super().__init__()
        
        hidden_dim = 128
        self.max_length = 50
        self.temperature = 1.0
        
        if config is not None:
            sg_params = config.get('sequence_generation', {})
            hidden_dim = sg_params.get('hidden_dim', hidden_dim)
            self.max_length = sg_params.get('max_length', self.max_length)
            self.temperature = sg_params.get('temperature', self.temperature)
            
        # We use a GRU cell to generate sequences step-by-step
        # Input to GRU at each step: previous coordinate (3)
        # Hidden state: context vector
        self.gru = nn.GRUCell(input_size=3, hidden_size=hidden_dim)
        
        # Maps initial context from the Backward Diffusion model to GRU hidden state
        self.context_proj = nn.Linear(in_context_dim, hidden_dim)
        
        # Predicts relative coordinates (dx, dy, dz)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Predicts node type probabilities (e.g., 0: continue, 1: branchpoint, 2: terminate)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3) 
        )

    def forward(self, context_embedding, start_coord, seq_len=None):
        """
        Training forward pass.
        Args:
            context_embedding: (B, in_context_dim) embedding of the node we extend from
            start_coord: (B, 3) relative starting coordinates (e.g. 0,0,0 if extending relative to parent)
            seq_len: integer, number of steps to generate. Defaults to max_length.
            
        Returns:
            coords: (B, seq_len, 3) generated relative coordinates
            types: (B, seq_len, 3) logits for node type
        """
        B = context_embedding.size(0)
        device = context_embedding.device
        
        if seq_len is None:
            seq_len = self.max_length
            
        h = self.context_proj(context_embedding)
        curr_x = start_coord
        
        coords = []
        types = []
        
        for _ in range(seq_len):
            h = self.gru(curr_x, h)
            
            # Predict delta coordinates and add to current
            dx = self.coord_head(h)
            curr_x = curr_x + dx
            coords.append(curr_x.unsqueeze(1))
            
            # Predict node type
            type_logits = self.type_head(h)
            types.append(type_logits.unsqueeze(1))
            
        coords_out = torch.cat(coords, dim=1)
        types_out = torch.cat(types, dim=1)
        
        return coords_out, types_out
        
    def sample(self, context_embedding, start_coord):
        """
        Inference loop with temperature sampling and termination logic.
        """
        B = context_embedding.size(0)
        device = context_embedding.device
        h = self.context_proj(context_embedding)
        curr_x = start_coord
        
        generated_coords = []
        generated_types = []
        
        # For batch inference, keep track of which sequences have terminated
        active_mask = torch.ones(B, dtype=torch.bool, device=device)
        
        for _ in range(self.max_length):
            if not active_mask.any():
                break
                
            h = self.gru(curr_x, h)
            dx = self.coord_head(h)
            
            # Apply temperature to coordinate variance if making probabilistic 
            # (Here we just do deterministic mean prediction for simplicity, 
            # but temperature could scale a VAE/diffusion noise addition)
            curr_x = curr_x + dx * self.temperature
            
            type_logits = self.type_head(h)
            type_probs = torch.softmax(type_logits / self.temperature, dim=-1)
            
            # Sample type: 0=continue, 1=branch, 2=terminate
            sampled_type = torch.multinomial(type_probs, 1).squeeze(-1)
            
            generated_coords.append(curr_x.clone().unsqueeze(1))
            generated_types.append(sampled_type.unsqueeze(1))
            
            # Update mask (if a node terminates, it is no longer active)
            active_mask = active_mask & (sampled_type != 2)
            
        if len(generated_coords) == 0:
            return torch.empty(B, 0, 3, device=device), torch.empty(B, 0, dtype=torch.long, device=device)
            
        return torch.cat(generated_coords, dim=1), torch.cat(generated_types, dim=1)

if __name__ == '__main__':
    print("Testing Continuous 3D Sequence Generator...")
    
    config = None
    if os.path.exists('config.yaml'):
        config = load_config('config.yaml')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for parallelization")
    
    generator = SequenceGenerator(config=config).to(device)
    
    # Mock inputs: batch of 4 endpoints to extend from
    B = 4
    context_dim = 128
    mock_context = torch.rand(B, context_dim, device=device)
    start_coords = torch.zeros(B, 3, device=device)
    
    print("Running training forward pass...")
    coords, types = generator(mock_context, start_coords, seq_len=10)
    print(f"Generated Coords Shape: {coords.shape}")
    print(f"Generated Types Shape: {types.shape}")
    
    print("\nRunning inference sampling loop (with termination logic)...")
    sampled_coords, sampled_types = generator.sample(mock_context, start_coords)
    print(f"Sampled Coords Shape (up to max_length): {sampled_coords.shape}")
    print(f"Sampled Types Shape: {sampled_types.shape}")
