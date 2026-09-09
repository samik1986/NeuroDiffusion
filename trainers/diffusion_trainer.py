import torch

class DiffusionTrainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        
    def train_step(self, x, inv_features, batch_assignment, t_expanded, true_edge_index):
        self.optimizer.zero_grad()
        
        # Generate negative samples for balanced loss
        B = int(batch_assignment.max().item() + 1)
        num_pos = true_edge_index.shape[1]
        
        if num_pos > 0:
            neg_u = torch.randint(0, x.shape[0], (num_pos,), device=self.device)
            neg_v = torch.randint(0, x.shape[0], (num_pos,), device=self.device)
            neg_edge_index = torch.stack([neg_u, neg_v], dim=0)
            
            candidate_edges = torch.cat([true_edge_index, neg_edge_index], dim=1)
            true_edge_labels = torch.cat([torch.ones(num_pos, device=self.device), torch.zeros(num_pos, device=self.device)])
        else:
            candidate_edges = true_edge_index
            true_edge_labels = torch.ones(0, device=self.device)
            
        edge_logits, out_flat = self.model(x, inv_features, batch_assignment, t_expanded, candidate_edges)
        
        edge_loss = torch.tensor(0.0, device=self.device)
        if len(edge_logits) > 0:
            edge_loss = self.loss_fn.edge_loss_fn(edge_logits, true_edge_labels.float())
            
        (self.loss_fn.edge_weight * edge_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return edge_loss.item(), out_flat.detach()
        
    def val_step(self, x, inv_features, batch_assignment, t_expanded, true_edge_index):
        B = int(batch_assignment.max().item() + 1)
        num_pos = true_edge_index.shape[1]
        
        if num_pos > 0:
            neg_u = torch.randint(0, x.shape[0], (num_pos,), device=self.device)
            neg_v = torch.randint(0, x.shape[0], (num_pos,), device=self.device)
            neg_edge_index = torch.stack([neg_u, neg_v], dim=0)
            
            candidate_edges = torch.cat([true_edge_index, neg_edge_index], dim=1)
            true_edge_labels = torch.cat([torch.ones(num_pos, device=self.device), torch.zeros(num_pos, device=self.device)])
        else:
            candidate_edges = true_edge_index
            true_edge_labels = torch.ones(0, device=self.device)
            
        edge_logits, out_flat = self.model(x, inv_features, batch_assignment, t_expanded, candidate_edges)
        
        edge_loss = torch.tensor(0.0, device=self.device)
        if len(edge_logits) > 0:
            edge_loss = self.loss_fn.edge_loss_fn(edge_logits, true_edge_labels.float())
            
        return edge_loss.item(), out_flat.detach()
