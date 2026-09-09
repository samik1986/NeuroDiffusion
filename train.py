import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_config
from data_loader import get_dataloader
from forward_diffusion import DiscreteGraphDiffusion
from backward_diffusion import BackwardDiffusionModel
from sequence_generator import SequenceGenerator
from heuristic_evaluator import ValidityHeuristicEvaluator
from loss_functions import NeuroDiffusionLoss

from trainers.diffusion_trainer import DiffusionTrainer
from trainers.generator_trainer import GeneratorTrainer
from trainers.heuristic_trainer import HeuristicTrainer

def plot_tree_corruption(x, original_edges, corrupted_edges, save_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x_np = x.cpu().numpy()
    
    # Plot nodes
    ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], c='black', s=5, alpha=0.5)
    
    # Plot original edges (light grey)
    if original_edges.shape[1] > 0:
        orig_u, orig_v = original_edges.cpu().numpy()
        for i in range(len(orig_u)):
            ax.plot([x_np[orig_u[i], 0], x_np[orig_v[i], 0]], 
                    [x_np[orig_u[i], 1], x_np[orig_v[i], 1]], 
                    [x_np[orig_u[i], 2], x_np[orig_v[i], 2]], 
                    c='lightgrey', alpha=0.3, linewidth=1)
                
    # Plot corrupted edges (red)
    if corrupted_edges.shape[1] > 0:
        corr_u, corr_v = corrupted_edges.cpu().numpy()
        for i in range(len(corr_u)):
            ax.plot([x_np[corr_u[i], 0], x_np[corr_v[i], 0]], 
                    [x_np[corr_u[i], 1], x_np[corr_v[i], 1]], 
                    [x_np[corr_u[i], 2], x_np[corr_v[i], 2]], 
                    c='red', linewidth=2)
                    
    ax.set_title("Forward Diffusion Tree Corruption")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def train():
    print("Starting Training Pipeline...")
    
    config = load_config('config.yaml')
    train_params = config.get('training', {})
    epochs = train_params.get('epochs', 100)
    lr = train_params.get('learning_rate', 0.0001)
    wd = train_params.get('weight_decay', 1e-4)
    save_every = train_params.get('save_every', 1)
    
    out_cfg = config.get('output', {})
    base_dir = out_cfg.get('base_dir', 'output')
    ckpt_dir = os.path.join(base_dir, out_cfg.get('checkpoints_dir', 'checkpoints'))
    val_dir = os.path.join(base_dir, out_cfg.get('validation_dir', 'validation'))
    log_dir = os.path.join(base_dir, out_cfg.get('logs_dir', 'logs'))
    img_dir = os.path.join(base_dir, out_cfg.get('images_dir', 'corrupted_disjoint_trees'))
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    tb_dir = os.path.join(base_dir, 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloader(config)
    if train_loader is None or val_loader is None:
        print("No data found in directory. Aborting.")
        return
    
    # Initialize all models
    forward_diffusion = DiscreteGraphDiffusion(config=config)
    diff_model = BackwardDiffusionModel(config=config).to(device)
    gen_model = SequenceGenerator(config=config).to(device)
    eval_model = ValidityHeuristicEvaluator(config=config).to(device)
    loss_module = NeuroDiffusionLoss(config=config).to(device)
    
    # Initialize optimizers & schedulers
    diff_opt = optim.AdamW(diff_model.parameters(), lr=lr, weight_decay=wd)
    diff_sch = optim.lr_scheduler.CosineAnnealingLR(diff_opt, T_max=epochs)
    
    gen_opt = optim.AdamW(gen_model.parameters(), lr=lr, weight_decay=wd)
    gen_sch = optim.lr_scheduler.CosineAnnealingLR(gen_opt, T_max=epochs)
    
    eval_opt = optim.AdamW(eval_model.parameters(), lr=lr, weight_decay=wd)
    eval_sch = optim.lr_scheduler.CosineAnnealingLR(eval_opt, T_max=epochs)
    
    # Initialize trainers
    diff_trainer = DiffusionTrainer(diff_model, diff_opt, diff_sch, loss_module, device)
    gen_trainer = GeneratorTrainer(gen_model, gen_opt, gen_sch, loss_module, device)
    eval_trainer = HeuristicTrainer(eval_model, eval_opt, eval_sch, device)
    
    print(f"Beginning co-training for {epochs} epochs...")
    
    for epoch in range(epochs):
        diff_model.train()
        gen_model.train()
        eval_model.train()
        
        epoch_diff_loss = 0.0
        epoch_gen_coord = 0.0
        epoch_gen_type = 0.0
        epoch_eval_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            if batch is None:
                continue
                
            x = batch['x'].to(device, non_blocking=True)
            inv_features = batch['inv_features'].to(device, non_blocking=True)
            true_edge_index = batch['edge_index'].to(device, non_blocking=True)
            batch_assignment = batch['batch'].to(device, non_blocking=True)
            
            gt_coords = batch['gt_coords'].to(device, non_blocking=True)
            gt_types = batch['gt_types'].to(device, non_blocking=True)
            gt_mask = batch['gt_mask'].to(device, non_blocking=True)
            
            if true_edge_index.shape[1] == 0:
                continue
                
            B = int(batch_assignment.max().item() + 1)
            t_batch = torch.randint(0, forward_diffusion.num_timesteps, (1,), device=device)
            noisy_edge_index, keep_mask = forward_diffusion.forward_sample(true_edge_index, t_batch)
            
            if batch_idx == 0:
                save_path = os.path.join(img_dir, f"epoch_{epoch+1}_corruption.png")
                plot_tree_corruption(x, true_edge_index, noisy_edge_index, save_path)
            
            # --- 1. Train Backward Diffusion ---
            diff_loss, context_emb = diff_trainer.train_step(
                x, inv_features, batch_assignment, t_batch.expand(B), true_edge_index
            )
            epoch_diff_loss += diff_loss
            
            # Extract valid context (e.g. mean pooling per graph for simplicity of this architectural pipeline)
            # In a real scenario, this would be specific node pairs.
            graph_context = []
            for i in range(B):
                mask = (batch_assignment == i)
                if mask.any():
                    graph_context.append(context_emb[mask].mean(dim=0))
                else:
                    graph_context.append(torch.zeros(context_emb.shape[-1], device=device))
            graph_context = torch.stack(graph_context)
            
            # --- 2. Train Sequence Generator ---
            coord_loss, type_loss, pred_coords, pred_types, gen_t = gen_trainer.train_step(
                graph_context, gt_coords, gt_types, gt_mask, gt_mask
            )
            epoch_gen_coord += coord_loss
            epoch_gen_type += type_loss
            
            # --- 3. Train Heuristic Evaluator ---
            # Use same graph context for T1 and T2 for dummy training
            t1_context = graph_context
            t2_context = graph_context
            
            gt_types_one_hot = torch.nn.functional.one_hot(gt_types, num_classes=3).float()
            eval_loss = eval_trainer.train_step(
                t1_context, t2_context, 
                gt_coords, gt_types_one_hot,       # "Valid/True"
                pred_coords, pred_types,   # "Fake/Generated"
                gen_t
            )
            epoch_eval_loss += eval_loss
            
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Diffusion', diff_loss, global_step)
            writer.add_scalar('Loss/Gen_Coord', coord_loss, global_step)
            writer.add_scalar('Loss/Gen_Type', type_loss, global_step)
            writer.add_scalar('Loss/Heuristic', eval_loss, global_step)
            
        diff_sch.step()
        gen_sch.step()
        eval_sch.step()
        
        # Validation Loop
        diff_model.eval()
        gen_model.eval()
        eval_model.eval()
        
        val_diff_loss = 0.0
        val_gen_coord = 0.0
        val_gen_type = 0.0
        val_eval_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                if batch is None:
                    continue
                    
                x = batch['x'].to(device, non_blocking=True)
                inv_features = batch['inv_features'].to(device, non_blocking=True)
                true_edge_index = batch['edge_index'].to(device, non_blocking=True)
                batch_assignment = batch['batch'].to(device, non_blocking=True)
                
                gt_coords = batch['gt_coords'].to(device, non_blocking=True)
                gt_types = batch['gt_types'].to(device, non_blocking=True)
                gt_mask = batch['gt_mask'].to(device, non_blocking=True)
                
                if true_edge_index.shape[1] == 0:
                    continue
                    
                B = int(batch_assignment.max().item() + 1)
                t_batch = torch.randint(0, forward_diffusion.num_timesteps, (1,), device=device)
                
                if batch_idx == 0:
                    noisy_edge_index, _ = forward_diffusion.forward_sample(true_edge_index, t_batch)
                    save_path = os.path.join(img_dir, f"epoch_{epoch+1}_val_corruption.png")
                    plot_tree_corruption(x, true_edge_index, noisy_edge_index, save_path)
                    
                diff_loss, context_emb = diff_trainer.val_step(
                    x, inv_features, batch_assignment, t_batch.expand(B), true_edge_index
                )
                val_diff_loss += diff_loss
                
                graph_context = []
                for i in range(B):
                    mask = (batch_assignment == i)
                    if mask.any():
                        graph_context.append(context_emb[mask].mean(dim=0))
                    else:
                        graph_context.append(torch.zeros(context_emb.shape[-1], device=device))
                graph_context = torch.stack(graph_context)
                
                coord_loss, type_loss, pred_coords, pred_types, gen_t = gen_trainer.val_step(
                    graph_context, gt_coords, gt_types, gt_mask, gt_mask
                )
                val_gen_coord += coord_loss
                val_gen_type += type_loss
                
                gt_types_one_hot = torch.nn.functional.one_hot(gt_types, num_classes=3).float()
                eval_loss = eval_trainer.val_step(
                    graph_context, graph_context, 
                    gt_coords, gt_types_one_hot,       
                    pred_coords, pred_types,   
                    gen_t
                )
                val_eval_loss += eval_loss
                
        writer.add_scalar('Val_Loss/Diffusion', val_diff_loss/max(len(val_loader), 1), epoch)
        writer.add_scalar('Val_Loss/Gen_Coord', val_gen_coord/max(len(val_loader), 1), epoch)
        writer.add_scalar('Val_Loss/Gen_Type', val_gen_type/max(len(val_loader), 1), epoch)
        writer.add_scalar('Val_Loss/Heuristic', val_eval_loss/max(len(val_loader), 1), epoch)
        
        print(f"==> Epoch [{epoch+1}/{epochs}] Diff: {epoch_diff_loss/len(train_loader):.4f} (Val {val_diff_loss/max(len(val_loader), 1):.4f}) | "
              f"GenC: {epoch_gen_coord/len(train_loader):.4f} (Val {val_gen_coord/max(len(val_loader), 1):.4f}) | "
              f"GenT: {epoch_gen_type/len(train_loader):.4f} (Val {val_gen_type/max(len(val_loader), 1):.4f}) | "
              f"Eval: {epoch_eval_loss/len(train_loader):.4f} (Val {val_eval_loss/max(len(val_loader), 1):.4f})")
        
        if (epoch + 1) % save_every == 0:
            torch.save(diff_model.state_dict(), os.path.join(ckpt_dir, f"diff_epoch_{epoch+1}.pth"))
            torch.save(gen_model.state_dict(), os.path.join(ckpt_dir, f"gen_epoch_{epoch+1}.pth"))
            torch.save(eval_model.state_dict(), os.path.join(ckpt_dir, f"eval_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoints for epoch {epoch+1}")
            
    print("Training Complete!")

if __name__ == '__main__':
    train()
