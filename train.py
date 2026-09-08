import os
import torch
import torch.optim as optim

from utils import load_config
from data_loader import get_dataloader
from forward_diffusion import DiscreteGraphDiffusion
from backward_diffusion import BackwardDiffusionModel
from sequence_generator import SequenceGenerator
from loss_functions import NeuroDiffusionLoss

def train():
    print("Starting Training Pipeline...")
    
    # 1. Configuration
    config = load_config('config.yaml')
    
    # Extract training parameters
    train_params = config.get('training', {})
    epochs = train_params.get('epochs', 100)
    lr = train_params.get('learning_rate', 0.001)
    save_every = train_params.get('save_every', 10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Data Loader
    train_loader, val_loader = get_dataloader(config)
    
    if train_loader is None or val_loader is None:
        print("No data found in directory. Aborting.")
        return
    
    # 3. Initialize Models
    forward_diffusion = DiscreteGraphDiffusion(config=config)
    
    # We will train the Backward GNN
    # Note: In a full pipeline, we'd also co-train the SequenceGenerator and Heuristic Evaluator.
    # For this script, we'll focus on the core Diffusion Edge Reconstruction.
    model = BackwardDiffusionModel(config=config).to(device)
    loss_module = NeuroDiffusionLoss(config=config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 4. Training Loop
    print(f"Beginning training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
                
            x = batch['x'].to(device, non_blocking=True)
            inv_features = batch['inv_features'].to(device, non_blocking=True)
            true_edge_index = batch['edge_index'].to(device, non_blocking=True)
            batch_assignment = batch['batch'].to(device, non_blocking=True)
            
            if true_edge_index.shape[1] == 0:
                continue
                
            optimizer.zero_grad()
            
            B = int(batch_assignment.max().item() + 1)
            t_batch = torch.randint(0, forward_diffusion.num_timesteps, (1,), device=device)
            noisy_edge_index, keep_mask = forward_diffusion.forward_sample(true_edge_index, t_batch)
            
            t_expanded = t_batch.expand(B)
            pred_edge_logits = model(x, inv_features, batch_assignment, t_expanded, true_edge_index)
            true_edge_labels = torch.ones(true_edge_index.shape[1], device=device)
            
            dummy_coords = torch.zeros(B, 1, 3, device=device)
            dummy_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            dummy_types = torch.zeros(B, 1, 3, device=device)
            dummy_true_types = torch.zeros(B, 1, dtype=torch.long, device=device)
            
            total_loss, loss_dict = loss_module(
                pred_edge_logits, true_edge_labels,
                dummy_coords, dummy_coords, dummy_mask,
                dummy_types, dummy_true_types, dummy_mask
            )
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += loss_dict['edge_loss']
                
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation Loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                    
                x = batch['x'].to(device, non_blocking=True)
                inv_features = batch['inv_features'].to(device, non_blocking=True)
                true_edge_index = batch['edge_index'].to(device, non_blocking=True)
                batch_assignment = batch['batch'].to(device, non_blocking=True)
                
                if true_edge_index.shape[1] == 0:
                    continue
                    
                B = int(batch_assignment.max().item() + 1)
                t_batch = torch.randint(0, forward_diffusion.num_timesteps, (1,), device=device)
                t_expanded = t_batch.expand(B)
                
                pred_edge_logits = model(x, inv_features, batch_assignment, t_expanded, true_edge_index)
                true_edge_labels = torch.ones(true_edge_index.shape[1], device=device)
                
                dummy_coords = torch.zeros(B, 1, 3, device=device)
                dummy_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
                dummy_types = torch.zeros(B, 1, 3, device=device)
                dummy_true_types = torch.zeros(B, 1, dtype=torch.long, device=device)
                
                _, v_dict = loss_module(
                    pred_edge_logits, true_edge_labels,
                    dummy_coords, dummy_coords, dummy_mask,
                    dummy_types, dummy_true_types, dummy_mask
                )
                val_loss += v_dict['edge_loss']
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"==> Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f"backward_diffusion_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint: backward_diffusion_epoch_{epoch+1}.pth")
            
    print("Training Complete!")

if __name__ == '__main__':
    train()
