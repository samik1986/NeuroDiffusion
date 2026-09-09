import torch

class HeuristicTrainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
    def train_step(self, t1_context, t2_context, valid_coords, valid_types, fake_coords, fake_types, t):
        self.optimizer.zero_grad()
        
        valid_logits = self.model(t1_context, t2_context, valid_coords, valid_types, t)
        fake_logits = self.model(t1_context, t2_context, fake_coords, fake_types, t)
        
        logits = torch.cat([valid_logits, fake_logits], dim=0)
        labels = torch.cat([torch.ones_like(valid_logits), torch.zeros_like(fake_logits)], dim=0)
        
        loss = self.bce_loss(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def val_step(self, t1_context, t2_context, valid_coords, valid_types, fake_coords, fake_types, t):
        valid_logits = self.model(t1_context, t2_context, valid_coords, valid_types, t)
        fake_logits = self.model(t1_context, t2_context, fake_coords, fake_types, t)
        
        logits = torch.cat([valid_logits, fake_logits], dim=0)
        labels = torch.cat([torch.ones_like(valid_logits), torch.zeros_like(fake_logits)], dim=0)
        
        loss = self.bce_loss(logits, labels)
        return loss.item()
