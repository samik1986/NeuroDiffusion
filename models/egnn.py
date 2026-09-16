import torch
import torch.nn as nn

def scatter_add(src, index, dim_size):
    shape = list(src.shape)
    shape[0] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(0, index_expanded, src)

class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim=0, act_fn=nn.SiLU(), coords_weight=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coords_weight = coords_weight
        
        # phi_e: edge message MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim * 2),
            act_fn,
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn
        )
        
        # phi_x: coordinate update MLP
        # Maps from edge message to a scalar weight for the coordinate difference
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1)
        )
        
        # phi_h: node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            act_fn,
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        h: (N, hidden_dim)
        x: (N, 3)
        edge_index: (2, E)
        edge_attr: (E, edge_dim)
        """
        row, col = edge_index
        
        # Compute distances
        coord_diff = x[row] - x[col] # (E, 3)
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1) # (E, 1)
        
        # Compute edge messages
        if edge_attr is not None:
            edge_input = torch.cat([h[row], h[col], radial, edge_attr], dim=1)
        else:
            edge_input = torch.cat([h[row], h[col], radial], dim=1)
            
        m_ij = self.edge_mlp(edge_input) # (E, hidden_dim)
        
        # Update coordinates (equivariant)
        # BOUND coord_weight to avoid exponential explosion
        coord_weight = torch.tanh(self.coord_mlp(m_ij)) * 10.0 # (E, 1)
        
        # Normalize coord_diff by distance to prevent scale explosions
        # Standard EGNN trick: trans = (coord_diff / (dist + 1e-8)) * coord_weight
        dist = torch.sqrt(radial + 1e-8)
        trans = (coord_diff / dist) * coord_weight # (E, 3)
        
        agg_trans = scatter_add(trans, row, dim_size=x.size(0))
        
        # Divide by degree to prevent density-based explosions
        degree = scatter_add(torch.ones_like(coord_weight), row, dim_size=x.size(0))
        agg_trans = agg_trans / (degree + 1e-8)
        
        x_out = x + agg_trans * self.coords_weight
        
        # Update node features (invariant)
        m_i = scatter_add(m_ij, row, dim_size=h.size(0))
        node_input = torch.cat([h, m_i], dim=1)
        h_out = h + self.node_mlp(node_input)
        
        return h_out, x_out

class EGNN(nn.Module):
    def __init__(self, in_node_dim, hidden_dim, out_node_dim, num_layers=4, edge_dim=0):
        super().__init__()
        self.embedding = nn.Linear(in_node_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EGNNLayer(hidden_dim, edge_dim=edge_dim))
            
        self.out_projection = nn.Linear(hidden_dim, out_node_dim)

    def forward(self, h, x, edge_index, edge_attr=None):
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        h = self.out_projection(h)
        return h, x
