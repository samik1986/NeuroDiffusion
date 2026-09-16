import torch

class DiffusionTrainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        
    def train_step(self, x, noisy_edge_index, batch_assignment, t_expanded, positive_edges):
        self.optimizer.zero_grad()
        
        # Generate negative samples for balanced loss
        B = int(batch_assignment.max().item() + 1)
        num_pos = positive_edges.shape[1]
        
        if num_pos > 0:
            # Intra-batch negative sampling
            counts = torch.bincount(batch_assignment)
            cum_counts = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), torch.cumsum(counts, dim=0)])
            
            neg_u = torch.randint(0, x.shape[0], (num_pos,), device=self.device)
            b = batch_assignment[neg_u]
            
            v_offset = (torch.rand(num_pos, device=self.device) * counts[b]).long()
            neg_v = cum_counts[b] + v_offset
            
            neg_edge_index = torch.stack([neg_u, neg_v], dim=0)
            
            candidate_edges = torch.cat([positive_edges, neg_edge_index], dim=1)
            true_edge_labels = torch.cat([torch.ones(num_pos, device=self.device), torch.zeros(num_pos, device=self.device)])
        else:
            candidate_edges = positive_edges
            true_edge_labels = torch.ones(0, device=self.device)
            
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            edge_logits, out_flat = self.model(x, noisy_edge_index, batch_assignment, t_expanded, candidate_edges)
            
            edge_loss = torch.tensor(0.0, device=self.device)
            if len(edge_logits) > 0:
                edge_loss = self.loss_fn.edge_loss_fn(edge_logits, true_edge_labels.float())
                
                # --- Toplogical Cycle Penalty (Optimized for Speed) ---
                pred_probs = torch.sigmoid(edge_logits)
                N = x.shape[0]
                
                # A perfect tree has N - B edges. We penalize generating more edges than necessary.
                noisy_edges = noisy_edge_index.shape[1] / 2.0
                total_edges = noisy_edges + (pred_probs.sum() / 2.0)
                target_edges = N - B
                
                tree_size_loss = torch.nn.functional.mse_loss(
                    total_edges.to(torch.float32), 
                    torch.tensor(target_edges, device=self.device, dtype=torch.float32)
                )
                
                tree_weight = 0.5
                
                loss = self.loss_fn.edge_weight * edge_loss + tree_weight * tree_size_loss
            else:
                # Dummy loss to prevent DDP deadlock when candidate edges are empty
                # Connects the loss to the model's output so backward() syncs properly inside DDP
                loss = out_flat.sum() * 0.0
            
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return edge_loss.item(), out_flat.detach()
        
    def val_step(self, x, noisy_edge_index, batch_assignment, t_expanded, positive_edges):
        B = int(batch_assignment.max().item() + 1)
        num_pos = positive_edges.shape[1]
        
        if num_pos > 0:
            counts = torch.bincount(batch_assignment)
            cum_counts = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), torch.cumsum(counts, dim=0)])
            
            neg_u = torch.randint(0, x.shape[0], (num_pos,), device=self.device)
            b = batch_assignment[neg_u]
            v_offset = (torch.rand(num_pos, device=self.device) * counts[b]).long()
            neg_v = cum_counts[b] + v_offset
            
            neg_edge_index = torch.stack([neg_u, neg_v], dim=0)
            
            candidate_edges = torch.cat([positive_edges, neg_edge_index], dim=1)
            true_edge_labels = torch.cat([torch.ones(num_pos, device=self.device), torch.zeros(num_pos, device=self.device)])
        else:
            candidate_edges = positive_edges
            true_edge_labels = torch.ones(0, device=self.device)
            
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            edge_logits, out_flat = self.model(x, noisy_edge_index, batch_assignment, t_expanded, candidate_edges)
            
            edge_loss = torch.tensor(0.0, device=self.device)
            if len(edge_logits) > 0:
                edge_loss = self.loss_fn.edge_loss_fn(edge_logits, true_edge_labels.float())
                
                # --- Toplogical Cycle Penalty (Optimized for Speed) ---
                pred_probs = torch.sigmoid(edge_logits)
                N = x.shape[0]
                
                noisy_edges = noisy_edge_index.shape[1] / 2.0
                total_edges = noisy_edges + (pred_probs.sum() / 2.0)
                target_edges = N - B
                
                tree_size_loss = torch.nn.functional.mse_loss(
                    total_edges.to(torch.float32), 
                    torch.tensor(target_edges, device=self.device, dtype=torch.float32)
                )
                
                tree_weight = 0.5
                
                loss = self.loss_fn.edge_weight * edge_loss + tree_weight * tree_size_loss
            else:
                loss = torch.tensor(0.0, device=self.device)
                
        return loss.item(), out_flat.detach(), candidate_edges.detach(), edge_logits.detach() if len(edge_logits) > 0 else edge_logits
