import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
import numpy as np

def plot_scatter_fast(ax, x_np, c='black', s=5, alpha=0.5):
    if len(x_np) > 1000:
        idx = np.random.choice(len(x_np), 1000, replace=False)
        x_plot = x_np[idx]
    else:
        x_plot = x_np
    ax.scatter(x_plot[:, 0], x_plot[:, 1], x_plot[:, 2], c=c, s=s, alpha=alpha)

def plot_edges_fast(ax, x_np, u, v, c, alpha=1.0, linewidth=1.0):
    if len(u) == 0: return
    if len(u) > 1000:
        idx = np.random.choice(len(u), 1000, replace=False)
        u = u[idx]
        v = v[idx]
    x_lines = np.empty(len(u) * 3)
    x_lines[0::3] = x_np[u, 0]
    x_lines[1::3] = x_np[v, 0]
    x_lines[2::3] = np.nan
    y_lines = np.empty(len(u) * 3)
    y_lines[0::3] = x_np[u, 1]
    y_lines[1::3] = x_np[v, 1]
    y_lines[2::3] = np.nan
    z_lines = np.empty(len(u) * 3)
    z_lines[0::3] = x_np[u, 2]
    z_lines[1::3] = x_np[v, 2]
    z_lines[2::3] = np.nan
    ax.plot(x_lines, y_lines, z_lines, c=c, alpha=alpha, linewidth=linewidth)

def plot_tree_corruption(x, original_edges, corrupted_edges, save_path):
    fig = plt.figure(figsize=(20, 10))
    x_np = x.cpu().numpy()
    
    # --- Plot 1: Original Tree ---
    ax1 = fig.add_subplot(121, projection='3d')
    plot_scatter_fast(ax1, x_np, c='black', s=5, alpha=0.5)
    
    if original_edges.shape[1] > 0:
        orig_u, orig_v = original_edges.cpu().numpy()
        plot_edges_fast(ax1, x_np, orig_u, orig_v, c='lightgrey', alpha=0.8, linewidth=1.5)
    ax1.set_title("Original Neuronal Tree")
    
    # --- Plot 2: Corrupted Tree ---
    ax2 = fig.add_subplot(122, projection='3d')
    plot_scatter_fast(ax2, x_np, c='black', s=5, alpha=0.5)
    
    # Plot original edges faintly in background
    if original_edges.shape[1] > 0:
        orig_u, orig_v = original_edges.cpu().numpy()
        plot_edges_fast(ax2, x_np, orig_u, orig_v, c='lightgrey', alpha=0.2, linewidth=1.0)
                     
    # Plot corrupted edges (red)
    if corrupted_edges.shape[1] > 0:
        corr_u, corr_v = corrupted_edges.cpu().numpy()
        plot_edges_fast(ax2, x_np, corr_u, corr_v, c='red', linewidth=2.0)
                     
    ax2.set_title("Forward Diffusion Tree Corruption")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_val_predictions(x, corrupted_edges, predicted_edges, save_path):
    fig = plt.figure(figsize=(20, 10))
    x_np = x.cpu().numpy()
    
    # --- Plot 1: Corrupted Tree ---
    ax1 = fig.add_subplot(121, projection='3d')
    plot_scatter_fast(ax1, x_np, c='black', s=5, alpha=0.5)
    
    if corrupted_edges.shape[1] > 0:
        corr_u, corr_v = corrupted_edges.cpu().numpy()
        plot_edges_fast(ax1, x_np, corr_u, corr_v, c='red', linewidth=2.0)
    ax1.set_title("Corrupted Val Graph")
    
    # --- Plot 2: Generated Graph ---
    ax2 = fig.add_subplot(122, projection='3d')
    plot_scatter_fast(ax2, x_np, c='black', s=5, alpha=0.5)
    
    # Plot surviving parts (faintly)
    if corrupted_edges.shape[1] > 0:
        corr_u, corr_v = corrupted_edges.cpu().numpy()
        plot_edges_fast(ax2, x_np, corr_u, corr_v, c='grey', alpha=0.5, linewidth=1.5)
                     
    # Plot generated missing edges (blue)
    if predicted_edges.shape[1] > 0:
        pred_u, pred_v = predicted_edges.cpu().numpy()
        plot_edges_fast(ax2, x_np, pred_u, pred_v, c='blue', linewidth=2.0)
                     
    ax2.set_title("Model Predicted Missing Edges")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_full_diffusion_process(x, true_edges, forward_diffusion, gen_model, t1_context, t2_context, save_path):
    # 2 rows, 5 columns (t=0, 25, 50, 75, 99)
    fig = plt.figure(figsize=(25, 10))
    x_np = x.cpu().numpy()
    
    timesteps = [0, 24, 49, 74, 99]
    
    # ROW 1: Forward Diffusion (Edges)
    for idx, t in enumerate(timesteps):
        ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
        plot_scatter_fast(ax, x_np, c='black', s=5, alpha=0.5)
        
        noisy_edges, _ = forward_diffusion.forward_sample(true_edges, t)
        
        if noisy_edges.shape[1] > 0:
            corr_u, corr_v = noisy_edges.cpu().numpy()
            plot_edges_fast(ax, x_np, corr_u, corr_v, c='red', linewidth=1.0)
                        
        ax.set_title(f"Forward (t={t})")
        
    # ROW 2: Backward Diffusion (Coordinates)
    # We sample a sequence of length 20 just for visual proxy
    if hasattr(gen_model, 'module'): gen_model.module.max_length = 20
    else: gen_model.max_length = 20
    final_coords, _, history = (gen_model.module if hasattr(gen_model, 'module') else gen_model).sample(t1_context[0:1], t2_context[0:1], num_timesteps=100)
    
    # history saves at 99, 75, 50, 25, 0 (reversed)
    # so we reverse it again to match left-to-right (t=99 on left? No, t=99 is pure noise, we want t=99 on left to show generation)
    # Wait, Forward goes t=0 -> 99 (clean -> noisy).
    # Backward goes t=99 -> 0 (noisy -> clean).
    
    for idx, h in enumerate(history):
        ax = fig.add_subplot(2, 5, 5 + idx + 1, projection='3d')
        
        # 1. Plot the original tree as faint background context
        plot_scatter_fast(ax, x_np, c='black', s=5, alpha=0.1)
        if true_edges.shape[1] > 0:
            orig_u, orig_v = true_edges.cpu().numpy()
            plot_edges_fast(ax, x_np, orig_u, orig_v, c='lightgrey', alpha=0.1, linewidth=1.0)
                        
        # 2. Plot the generating sequence
        gen_x = h['coords'][0].cpu().numpy()
        ax.scatter(gen_x[:, 0], gen_x[:, 1], gen_x[:, 2], c='blue', s=20, alpha=0.8)
        
        # Connect the generated sequence path
        if len(gen_x) > 1:
            ax.plot(gen_x[:, 0], gen_x[:, 1], gen_x[:, 2], c='blue', linewidth=2)
                    
        ax.set_title(f"Backward (t={h['t']})")
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main_worker(rank, world_size, config):
    print(f"Starting Training Pipeline on rank {rank}...")
    
    # Init DDP
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
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
    
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        tb_dir = os.path.join(base_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None
        
    # Make sure dirs are created before other ranks proceed
    dist.barrier()

    
    
    
    train_loader, val_loader = get_dataloader(config, world_size, rank)
    if train_loader is None or val_loader is None:
        print("No data found in directory. Aborting.")
        return
    
    # Initialize all models
    forward_diffusion = DiscreteGraphDiffusion(config=config)
    diff_model = BackwardDiffusionModel(config=config).to(device)
    gen_model = SequenceGenerator(config=config).to(device)
    eval_model = ValidityHeuristicEvaluator(config=config).to(device)
    loss_module = NeuroDiffusionLoss(config=config).to(device)
    # Wrap in DDP
    diff_model = DDP(diff_model, device_ids=[rank], find_unused_parameters=True)
    gen_model = DDP(gen_model, device_ids=[rank], find_unused_parameters=True)
    eval_model = DDP(eval_model, device_ids=[rank], find_unused_parameters=True)
    
    
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
    
    if rank == 0:
        print(f"Beginning co-training for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        diff_model.train()
        gen_model.train()
        eval_model.train()
        
        epoch_diff_loss = 0.0
        epoch_gen_coord = 0.0
        epoch_gen_type = 0.0
        epoch_eval_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            skip_batch = torch.tensor(1 if batch is None else 0, device=device)
            dist.all_reduce(skip_batch, op=dist.ReduceOp.SUM)
            if skip_batch.item() > 0:
                continue
                
            x = batch['x'].to(device, non_blocking=True)
            inv_features = batch['inv_features'].to(device, non_blocking=True)
            true_edge_index = batch['edge_index'].to(device, non_blocking=True)
            batch_assignment = batch['batch'].to(device, non_blocking=True)
            
            gt_coords = batch['gt_coords'].to(device, non_blocking=True)
            gt_types = batch['gt_types'].to(device, non_blocking=True)
            gt_mask = batch['gt_mask'].to(device, non_blocking=True)
            
            skip_edges = torch.tensor(1 if true_edge_index.shape[1] == 0 else 0, device=device)
            dist.all_reduce(skip_edges, op=dist.ReduceOp.SUM)
            if skip_edges.item() > 0:
                continue
                
            B = int(batch_assignment.max().item() + 1)
            t_batch = torch.randint(0, forward_diffusion.num_timesteps, (1,), device=device)
            noisy_edge_index, keep_mask = forward_diffusion.forward_sample(true_edge_index, t_batch)
            
            if batch_idx == 0 and rank == 0:
                mask = (batch_assignment == 0)
                node_indices = torch.where(mask)[0]
                if len(node_indices) > 0:
                    x_single = x[mask]
                    edge_mask_orig = mask[true_edge_index[0]] & mask[true_edge_index[1]]
                    true_edges_single = true_edge_index[:, edge_mask_orig]
                    edge_mask_corr = mask[noisy_edge_index[0]] & mask[noisy_edge_index[1]]
                    noisy_edges_single = noisy_edge_index[:, edge_mask_corr]
                    mapping = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                    mapping[node_indices] = torch.arange(len(node_indices), device=device)
                    true_edges_single = mapping[true_edges_single]
                    noisy_edges_single = mapping[noisy_edges_single]
                    
                    save_path = os.path.join(img_dir, f"epoch_{epoch+1}_corruption.png")
                    plot_tree_corruption(x_single, true_edges_single, noisy_edges_single, save_path)
            
            # --- 1. Train Backward Diffusion ---
            diff_loss, context_emb = diff_trainer.train_step(
                x, noisy_edge_index, batch_assignment, t_batch.expand(B), true_edge_index
            )
            epoch_diff_loss += diff_loss
            
            # Extract valid context (e.g. mean pooling per graph for simplicity of this architectural pipeline)
            # In a real scenario, this would be specific node pairs.
            # Vectorized scatter mean for context extraction
            counts = torch.bincount(batch_assignment, minlength=B).clamp(min=1).unsqueeze(1).float()
            graph_context = torch.zeros(B, context_emb.size(1), device=device, dtype=context_emb.dtype)
            graph_context.scatter_add_(0, batch_assignment.unsqueeze(1).expand_as(context_emb), context_emb)
            graph_context = graph_context / counts
            
            # Use same graph context for T1 and T2 for dummy training (simulating pairwise generation)
            t1_context = graph_context
            t2_context = graph_context
            
            coord_loss, type_loss, pred_coords, pred_types, gen_t = gen_trainer.train_step(
                t1_context, t2_context, gt_coords, gt_types, gt_mask, gt_mask
            )
            epoch_gen_coord += coord_loss
            epoch_gen_type += type_loss
            
            # --- 3. Train Heuristic Evaluator ---
            
            gt_types_one_hot = torch.nn.functional.one_hot(gt_types, num_classes=3).float()
            eval_loss = eval_trainer.train_step(
                t1_context, t2_context, 
                gt_coords, gt_types_one_hot,       # "Valid/True"
                pred_coords, pred_types,   # "Fake/Generated"
                gen_t
            )
            epoch_eval_loss += eval_loss
            
            global_step = epoch * len(train_loader) + batch_idx
            if rank == 0:
                writer.add_scalar('Loss/Diffusion', diff_loss, global_step)
            if rank == 0:
                writer.add_scalar('Loss/Gen_Coord', coord_loss, global_step)
            if rank == 0:
                writer.add_scalar('Loss/Gen_Type', type_loss, global_step)
            if rank == 0:
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
            for batch_idx, batch in enumerate(tqdm(val_loader, disable=(rank!=0), desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                skip_batch = torch.tensor(1 if batch is None else 0, device=device)
                dist.all_reduce(skip_batch, op=dist.ReduceOp.SUM)
                if skip_batch.item() > 0:
                    continue
                    
                x = batch['x'].to(device, non_blocking=True)
                inv_features = batch['inv_features'].to(device, non_blocking=True)
                true_edge_index = batch['edge_index'].to(device, non_blocking=True)
                batch_assignment = batch['batch'].to(device, non_blocking=True)
                
                gt_coords = batch['gt_coords'].to(device, non_blocking=True)
                gt_types = batch['gt_types'].to(device, non_blocking=True)
                gt_mask = batch['gt_mask'].to(device, non_blocking=True)
                
                skip_edges = torch.tensor(1 if true_edge_index.shape[1] == 0 else 0, device=device)
                dist.all_reduce(skip_edges, op=dist.ReduceOp.SUM)
                if skip_edges.item() > 0:
                    continue
                    
                B = int(batch_assignment.max().item() + 1)
                t_batch = torch.randint(0, forward_diffusion.num_timesteps, (1,), device=device)
                
                noisy_edge_index, _ = forward_diffusion.forward_sample(true_edge_index, t_batch)
                
                if batch_idx == 0 and rank == 0:
                    
                    mask = (batch_assignment == 0)
                    node_indices = torch.where(mask)[0]
                    if len(node_indices) > 0:
                        x_single = x[mask]
                        edge_mask_orig = mask[true_edge_index[0]] & mask[true_edge_index[1]]
                        true_edges_single = true_edge_index[:, edge_mask_orig]
                        edge_mask_corr = mask[noisy_edge_index[0]] & mask[noisy_edge_index[1]]
                        noisy_edges_single = noisy_edge_index[:, edge_mask_corr]
                        mapping = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                        mapping[node_indices] = torch.arange(len(node_indices), device=device)
                        true_edges_single = mapping[true_edges_single]
                        noisy_edges_single = mapping[noisy_edges_single]
                        
                        save_path = os.path.join(img_dir, f"epoch_{epoch+1}_val_corruption.png")
                        plot_tree_corruption(x_single, true_edges_single, noisy_edges_single, save_path)
                    
                diff_loss, context_emb, cand_edges, edge_logits = diff_trainer.val_step(
                    x, noisy_edge_index, batch_assignment, t_batch.expand(B), true_edge_index
                )
                val_diff_loss += diff_loss
                
                if batch_idx == 0 and rank == 0:
                    # Filter candidate edges that were predicted as positive
                    if len(edge_logits) > 0:
                        pred_mask = torch.sigmoid(edge_logits) > 0.5
                        predicted_edges = cand_edges[:, pred_mask]
                        
                        # Filter to just the single graph we are plotting
                        pred_mask_single = mask[predicted_edges[0]] & mask[predicted_edges[1]]
                        predicted_edges_single = predicted_edges[:, pred_mask_single]
                        predicted_edges_single = mapping[predicted_edges_single]
                        
                        save_path_gen = os.path.join(img_dir, f"epoch_{epoch+1}_val_gen.png")
                        plot_val_predictions(x_single, noisy_edges_single, predicted_edges_single, save_path_gen)

                
                counts = torch.bincount(batch_assignment, minlength=B).clamp(min=1).unsqueeze(1).float()
                graph_context = torch.zeros(B, context_emb.size(1), device=device)
                graph_context.scatter_add_(0, batch_assignment.unsqueeze(1).expand_as(context_emb), context_emb)
                graph_context = graph_context / counts
                
                # Use same graph context for T1 and T2 for dummy training
                t1_context = graph_context
                t2_context = graph_context
                
                coord_loss, type_loss, pred_coords, pred_types, gen_t = gen_trainer.val_step(
                    t1_context, t2_context, gt_coords, gt_types, gt_mask, gt_mask
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
                
                # Plot full gradual process on the last batch of validation
                if batch_idx == len(val_loader) - 1 and rank == 0:
                    save_path_full = os.path.join(img_dir, f"epoch_{epoch+1}_full_diffusion.png")
                    
                    # Extract single graph for the forward corruption visualization
                    mask = (batch_assignment == 0)
                    node_indices = torch.where(mask)[0]
                    if len(node_indices) > 0:
                        x_single = x[mask]
                        edge_mask_orig = mask[true_edge_index[0]] & mask[true_edge_index[1]]
                        true_edges_single = true_edge_index[:, edge_mask_orig]
                        mapping = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                        mapping[node_indices] = torch.arange(len(node_indices), device=device)
                        true_edges_single = mapping[true_edges_single]
                        
                        plot_full_diffusion_process(
                            x_single, true_edges_single, forward_diffusion, gen_model, 
                            t1_context, t2_context, save_path_full
                        )
                
        if rank == 0:
            writer.add_scalar('Val_Loss/Diffusion', val_diff_loss/max(len(val_loader), 1), epoch)
        if rank == 0:
            writer.add_scalar('Val_Loss/Gen_Coord', val_gen_coord/max(len(val_loader), 1), epoch)
        if rank == 0:
            writer.add_scalar('Val_Loss/Gen_Type', val_gen_type/max(len(val_loader), 1), epoch)
        if rank == 0:
            writer.add_scalar('Val_Loss/Heuristic', val_eval_loss/max(len(val_loader), 1), epoch)
        
        print(f"==> Epoch [{epoch+1}/{epochs}] Diff: {epoch_diff_loss/len(train_loader):.4f} (Val {val_diff_loss/max(len(val_loader), 1):.4f}) | "
              f"GenC: {epoch_gen_coord/len(train_loader):.4f} (Val {val_gen_coord/max(len(val_loader), 1):.4f}) | "
              f"GenT: {epoch_gen_type/len(train_loader):.4f} (Val {val_gen_type/max(len(val_loader), 1):.4f}) | "
              f"Eval: {epoch_eval_loss/len(train_loader):.4f} (Val {val_eval_loss/max(len(val_loader), 1):.4f})")
        
        if (epoch + 1) % save_every == 0 and rank == 0:
            torch.save(diff_model.state_dict(), os.path.join(ckpt_dir, f"diff_epoch_{epoch+1}.pth"))
            torch.save(gen_model.state_dict(), os.path.join(ckpt_dir, f"gen_epoch_{epoch+1}.pth"))
            torch.save(eval_model.state_dict(), os.path.join(ckpt_dir, f"eval_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoints for epoch {epoch+1}")
            
    print("Training Complete!")



def train():
    import os
    config = load_config('config.yaml')
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        print(f"Spawning DDP across {world_size} GPUs...")
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, config))
    else:
        # Fallback to single process
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        main_worker(0, 1, config)
        
if __name__ == '__main__':
    train()

