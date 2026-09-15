import torch
import torch.nn.functional as F

class GeneratorTrainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, num_timesteps=100):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.num_timesteps = num_timesteps
        self.scaler = torch.amp.GradScaler('cuda')
        
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def train_step(self, t1_context, t2_context, true_coords, true_types, coord_mask, type_mask):
        B = t1_context.size(0)
        self.optimizer.zero_grad()
        
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(true_coords)
        
        a_cp = self.alphas_cumprod[t].view(B, 1, 1)
        noisy_coords = torch.sqrt(a_cp) * true_coords + torch.sqrt(1 - a_cp) * noise
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise_pred, pred_types = self.model(noisy_coords, t, t1_context, t2_context, padding_mask=coord_mask)
            
            coord_loss = torch.tensor(0.0, device=self.device)
            if coord_mask.any():
                raw_noise_loss = F.mse_loss(noise_pred, noise, reduction='none')
                masked_noise_loss = raw_noise_loss.mean(dim=-1) * coord_mask.float()
                coord_loss = masked_noise_loss.sum() / coord_mask.sum().clamp(min=1)
                
            type_loss = torch.tensor(0.0, device=self.device)
            if type_mask.any():
                flat_pred_types = pred_types.view(-1, 3)
                flat_true_types = true_types.view(-1)
                flat_mask = type_mask.view(-1)
                raw_type_loss = self.loss_fn.type_loss_fn(flat_pred_types, flat_true_types)
                masked_type_loss = raw_type_loss * flat_mask.float()
                type_loss = masked_type_loss.sum() / flat_mask.sum().clamp(min=1)
                
            total_loss = (self.loss_fn.coord_weight * coord_loss) + (self.loss_fn.type_weight * type_loss)
            
        if total_loss > 0:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        pred_x0 = (noisy_coords - torch.sqrt(1 - a_cp) * noise_pred) / torch.sqrt(a_cp)
        return coord_loss.item(), type_loss.item(), pred_x0.detach(), torch.softmax(pred_types.detach(), dim=-1), t

    def val_step(self, t1_context, t2_context, true_coords, true_types, coord_mask, type_mask):
        B = t1_context.size(0)
        
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(true_coords)
        
        a_cp = self.alphas_cumprod[t].view(B, 1, 1)
        noisy_coords = torch.sqrt(a_cp) * true_coords + torch.sqrt(1 - a_cp) * noise
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise_pred, pred_types = self.model(noisy_coords, t, t1_context, t2_context, padding_mask=coord_mask)
            
            coord_loss = torch.tensor(0.0, device=self.device)
            if coord_mask.any():
                raw_noise_loss = F.mse_loss(noise_pred, noise, reduction='none')
                masked_noise_loss = raw_noise_loss.mean(dim=-1) * coord_mask.float()
                coord_loss = masked_noise_loss.sum() / coord_mask.sum().clamp(min=1)
                
            type_loss = torch.tensor(0.0, device=self.device)
            if type_mask.any():
                flat_pred_types = pred_types.view(-1, 3)
                flat_true_types = true_types.view(-1)
                flat_mask = type_mask.view(-1)
                raw_type_loss = self.loss_fn.type_loss_fn(flat_pred_types, flat_true_types)
                masked_type_loss = raw_type_loss * flat_mask.float()
                type_loss = masked_type_loss.sum() / flat_mask.sum().clamp(min=1)
            
        pred_x0 = (noisy_coords - torch.sqrt(1 - a_cp) * noise_pred) / torch.sqrt(a_cp)
        return coord_loss.item(), type_loss.item(), pred_x0.detach(), torch.softmax(pred_types.detach(), dim=-1), t
