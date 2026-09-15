import torch
import math

class DiscreteGraphDiffusion:
    """
    Implements the discrete forward diffusion process for graph edges.
    Simulates the fragmentation of neurons by dropping edges over time.
    """
    def __init__(self, config=None, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear'):
        if config is not None:
            fd_params = config.get('forward_diffusion', {})
            num_timesteps = fd_params.get('num_timesteps', num_timesteps)
            beta_start = fd_params.get('beta_start', beta_start)
            beta_end = fd_params.get('beta_end', beta_end)
            schedule = fd_params.get('schedule', schedule)
            
        self.num_timesteps = num_timesteps
        
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # GPU compatibility
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        Provides a smoother transition for edge dropping.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
        
    def forward_sample(self, edge_index, t):
        """
        Samples the fragmented graph at timestep t by dropping edges.
        
        Args:
            edge_index: (2, E) tensor of true edges
            t: integer or scalar tensor representing the current timestep [0, num_timesteps-1]
            
        Returns:
            noisy_edge_index: The subset of edge_index kept at timestep t.
            edge_mask: Boolean mask of shape (E,) indicating which edges are kept.
        """
        if len(edge_index) == 0 or edge_index.shape[1] == 0:
            return edge_index, torch.ones(0, dtype=torch.bool, device=self.device)
            
        if isinstance(t, torch.Tensor):
            t = t.item()
            
        # Ensure input is on correct device
        edge_index = edge_index.to(self.device)
            
        # alpha_bar_t represents the cumulative probability of an edge NOT being dropped by step t
        alpha_bar_t = self.alphas_cumprod[t]
        
        num_edges = edge_index.shape[1]
        keep_probs = torch.full((num_edges,), alpha_bar_t.item(), device=self.device)
        
        # Sample keep mask
        random_uniform = torch.rand(num_edges, device=self.device)
        edge_mask = random_uniform < keep_probs
        
        noisy_edge_index = edge_index[:, edge_mask]
        return noisy_edge_index, edge_mask

if __name__ == '__main__':
    import os
    import sys
    
    # Load config if possible
    config = None
    if os.path.exists('config.yaml'):
        from utils import load_config
        config = load_config('config.yaml')
        print("Loaded config from config.yaml")

    print("Testing Discrete Forward Diffusion Process...")
    diffusion = DiscreteGraphDiffusion(config=config)
    print(f"Using device: {diffusion.device} for parallelization")
    
    # Mock an edge_index with 10 edges
    mock_edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    
    print(f"Original Edges: {mock_edges.shape[1]}")
    
    timesteps_to_test = [0, diffusion.num_timesteps//4, diffusion.num_timesteps//2, 3*diffusion.num_timesteps//4, diffusion.num_timesteps-1]
    
    for timestep in timesteps_to_test:
        noisy_edges, mask = diffusion.forward_sample(mock_edges, timestep)
        print(f"Timestep {timestep:02d} | Alpha_bar: {diffusion.alphas_cumprod[timestep]:.4f} | "
              f"Remaining Edges: {noisy_edges.shape[1]}/10")
