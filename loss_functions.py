import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import load_config

class NeuroDiffusionLoss(nn.Module):
    """
    Custom Loss Function for NeuroDiffusion.
    Combines:
    1. Edge BCE: Binary Cross Entropy for predicting missing edges across disjoint fragments.
    2. Coordinate MSE: Mean Squared Error for autoregressively generated 3D coordinates.
    3. Type CE: Cross Entropy for classification of sequence nodes (continue, branch, terminate).
    """
    def __init__(self, config=None):
        super().__init__()
        
        self.edge_weight = 1.0
        self.coord_weight = 10.0
        self.type_weight = 0.5
        
        if config is not None:
            lw_params = config.get('loss_weights', {})
            self.edge_weight = lw_params.get('edge_bce', self.edge_weight)
            self.coord_weight = lw_params.get('coord_mse', self.coord_weight)
            self.type_weight = lw_params.get('type_ce', self.type_weight)
            
        # Binary Cross Entropy with Logits for Edge Prediction
        # Uses pos_weight to handle class imbalance (many more disconnected pairs than connected)
        # We can dynamically pass pos_weight during forward, but BCEWithLogitsLoss is best here
        self.edge_loss_fn = nn.BCEWithLogitsLoss()
        
        # Mean Squared Error for Coordinates
        self.coord_loss_fn = nn.MSELoss(reduction='none') # Compute mask manually
        
        # Cross Entropy for Node Types (Continue=0, Branch=1, Terminate=2)
        self.type_loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, 
                pred_edge_logits, true_edge_labels,
                pred_coords, true_coords, coord_mask,
                pred_types, true_types, type_mask):
        """
        Args:
            pred_edge_logits: (E_cand,) edge prediction logits
            true_edge_labels: (E_cand,) binary labels (1=edge, 0=no edge)
            
            pred_coords: (B, seq_len, 3) predicted coordinates
            true_coords: (B, seq_len, 3) ground truth coordinates
            coord_mask: (B, seq_len) boolean mask where True indicates valid sequence items
            
            pred_types: (B, seq_len, 3) predicted node type logits
            true_types: (B, seq_len) ground truth node types (integers 0,1,2)
            type_mask: (B, seq_len) boolean mask for valid types
            
        Returns:
            total_loss: scalar combined loss
            loss_dict: dictionary tracking individual loss components
        """
        # 1. Edge Loss
        edge_loss = torch.tensor(0.0, device=pred_edge_logits.device)
        if len(pred_edge_logits) > 0:
            edge_loss = self.edge_loss_fn(pred_edge_logits, true_edge_labels.float())
            
        # 2. Coordinate MSE Loss (Masked)
        coord_loss = torch.tensor(0.0, device=pred_coords.device)
        if coord_mask.any():
            raw_coord_loss = self.coord_loss_fn(pred_coords, true_coords) # (B, seq_len, 3)
            # Average over dimensions, then apply mask
            masked_coord_loss = raw_coord_loss.mean(dim=-1) * coord_mask.float()
            coord_loss = masked_coord_loss.sum() / coord_mask.sum().clamp(min=1)
            
        # 3. Node Type CE Loss (Masked)
        type_loss = torch.tensor(0.0, device=pred_types.device)
        if type_mask.any():
            # CrossEntropy expects (N, C) for logits and (N) for targets
            flat_pred_types = pred_types.view(-1, 3)
            flat_true_types = true_types.view(-1)
            flat_mask = type_mask.view(-1)
            
            raw_type_loss = self.type_loss_fn(flat_pred_types, flat_true_types)
            masked_type_loss = raw_type_loss * flat_mask.float()
            type_loss = masked_type_loss.sum() / flat_mask.sum().clamp(min=1)
            
        # Combine Losses
        total_loss = (self.edge_weight * edge_loss + 
                      self.coord_weight * coord_loss + 
                      self.type_weight * type_loss)
                      
        loss_dict = {
            'total_loss': total_loss.item(),
            'edge_loss': edge_loss.item(),
            'coord_loss': coord_loss.item(),
            'type_loss': type_loss.item()
        }
        
        return total_loss, loss_dict

if __name__ == '__main__':
    print("Testing NeuroDiffusion Loss Module...")
    
    config = None
    if os.path.exists('config.yaml'):
        config = load_config('config.yaml')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_module = NeuroDiffusionLoss(config=config).to(device)
    
    print(f"Weights -> Edge: {loss_module.edge_weight}, Coord: {loss_module.coord_weight}, Type: {loss_module.type_weight}")
    
    # Mock Data
    pred_edge_logits = torch.randn(10, device=device)
    true_edge_labels = torch.randint(0, 2, (10,), device=device)
    
    B, seq_len = 2, 5
    pred_coords = torch.randn(B, seq_len, 3, device=device)
    true_coords = torch.randn(B, seq_len, 3, device=device)
    coord_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
    coord_mask[0, 3:] = False # Simulate terminated sequence
    
    pred_types = torch.randn(B, seq_len, 3, device=device)
    true_types = torch.randint(0, 3, (B, seq_len), device=device)
    type_mask = coord_mask.clone()
    
    total_loss, loss_dict = loss_module(
        pred_edge_logits, true_edge_labels,
        pred_coords, true_coords, coord_mask,
        pred_types, true_types, type_mask
    )
    
    print("\nComputed Losses:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
