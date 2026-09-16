import torch
import torch.nn.functional as F

class HeuristicTrainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mse_loss = torch.nn.MSELoss()
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        
    def train_step(self, t1_context, t2_context, valid_coords, valid_types, fake_coords, fake_types, t):
        self.optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Compute actual structural gap between generated and true path
            # MSE per sample in batch
            raw_mse = F.mse_loss(fake_coords, valid_coords, reduction='none').mean(dim=[1, 2])
            
            # Make it scale-invariant by dividing by the squared distance of the path endpoints
            # Adding epsilon to prevent division by zero
            scale = torch.norm(valid_coords[:, -1, :] - valid_coords[:, 0, :], dim=-1) ** 2 + 1e-6
            # Use log1p (log(1 + x)) to squash massive outliers and prevent the MSE loss from skewing
            true_gap_for_fake = torch.log1p(raw_mse / scale).detach()
            
            # Model predicts the gap for the fake sequence
            pred_gap_for_fake = self.model(t1_context, t2_context, fake_coords, fake_types, t).squeeze(-1)
            
            # Model predicts the gap for the real sequence (which should be 0)
            pred_gap_for_real = self.model(t1_context, t2_context, valid_coords, valid_types, t).squeeze(-1)
            true_gap_for_real = torch.zeros_like(pred_gap_for_real)
            
            # Regression loss: Predict the correct gap scalar
            loss_fake = self.mse_loss(pred_gap_for_fake, true_gap_for_fake)
            loss_real = self.mse_loss(pred_gap_for_real, true_gap_for_real)
            
            loss = loss_fake + loss_real
            
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
        
    def val_step(self, t1_context, t2_context, valid_coords, valid_types, fake_coords, fake_types, t):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw_mse = F.mse_loss(fake_coords, valid_coords, reduction='none').mean(dim=[1, 2])
            scale = torch.norm(valid_coords[:, -1, :] - valid_coords[:, 0, :], dim=-1) ** 2 + 1e-6
            true_gap_for_fake = torch.log1p(raw_mse / scale).detach()
            
            pred_gap_for_fake = self.model(t1_context, t2_context, fake_coords, fake_types, t).squeeze(-1)
            pred_gap_for_real = self.model(t1_context, t2_context, valid_coords, valid_types, t).squeeze(-1)
            true_gap_for_real = torch.zeros_like(pred_gap_for_real)
            
            loss_fake = self.mse_loss(pred_gap_for_fake, true_gap_for_fake)
            loss_real = self.mse_loss(pred_gap_for_real, true_gap_for_real)
            
            loss = loss_fake + loss_real
            
        return loss.item()
