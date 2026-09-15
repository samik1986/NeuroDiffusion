import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import get_dataloader
from utils.utils import load_config
from models.backward_diffusion import BackwardDiffusionModel
from models.sequence_generator import SequenceGenerator
from models.heuristic_evaluator import HeuristicEvaluator
from models.forward_diffusion import ForwardTreeDiffusion
from utils.loss_functions import NeuroDiffusionLoss
from trainers.diffusion_trainer import DiffusionTrainer
from trainers.generator_trainer import GeneratorTrainer
from trainers.evaluator_trainer import EvaluatorTrainer

def test_batch_size(batch_size, num_batches=5):
    print(f"\n--- Testing Batch Size: {batch_size} ---")
    config = load_config('config.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Overwrite config for testing
    config['dataloader']['batch_size'] = batch_size
    config['dataloader']['num_workers'] = 4
    
    train_loader = get_dataloader(config['dataloader']['swc_dir'], batch_size=batch_size, num_workers=4)
    
    diff_model = BackwardDiffusionModel(config=config).to(device)
    gen_model = SequenceGenerator(config=config).to(device)
    eval_model = ValidityHeuristicEvaluator(config=config).to(device)
    
    diff_opt = torch.optim.Adam(diff_model.parameters(), lr=1e-4)
    gen_opt = torch.optim.Adam(gen_model.parameters(), lr=1e-4)
    eval_opt = torch.optim.Adam(eval_model.parameters(), lr=1e-4)
    
    loss_fn = NeuroDiffusionLoss()
    
    diff_trainer = DiffusionTrainer(diff_model, diff_opt, None, loss_fn, device)
    gen_trainer = GeneratorTrainer(gen_model, gen_opt, None, loss_fn, device)
    eval_trainer = HeuristicTrainer(eval_model, eval_opt, None, loss_fn, device)
    
    forward_diffusion = DiscreteGraphDiffusion(config=config)
    
    total_time = 0.0
    processed_batches = 0
    
    try:
        start_loader = time.time()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            if batch is None:
                continue
                
            x = batch['x'].to(device)
            inv_features = batch['inv_features'].to(device)
            true_edge_index = batch['edge_index'].to(device)
            batch_assignment = batch['batch'].to(device)
            
            gt_coords = batch['gt_coords'].to(device)
            gt_types = batch['gt_types'].to(device)
            gt_mask = batch['gt_mask'].to(device)
            
            if true_edge_index.shape[1] == 0:
                continue
                
            B = int(batch_assignment.max().item() + 1)
            t_batch = torch.randint(0, forward_diffusion.num_timesteps, (1,), device=device)
            
            start_batch = time.time()
            
            # Step 1
            diff_loss, context_emb = diff_trainer.train_step(
                x, inv_features, batch_assignment, t_batch.expand(B), true_edge_index
            )
            
            # Step 2
            graph_context = []
            for i in range(B):
                mask = (batch_assignment == i)
                if mask.any():
                    graph_context.append(context_emb[mask].mean(dim=0))
                else:
                    graph_context.append(torch.zeros(context_emb.shape[-1], device=device))
            graph_context = torch.stack(graph_context)
            
            t1_context = graph_context
            t2_context = graph_context
            
            coord_loss, type_loss, pred_coords, pred_types, gen_t = gen_trainer.train_step(
                t1_context, t2_context, gt_coords, gt_types, gt_mask, gt_mask
            )
            
            # Step 3
            gt_types_one_hot = torch.nn.functional.one_hot(gt_types, num_classes=3).float()
            eval_loss = eval_trainer.train_step(
                t1_context, t2_context, gt_coords, gt_types_one_hot, pred_coords, pred_types, gen_t
            )
            
            end_batch = time.time()
            total_time += (end_batch - start_batch)
            processed_batches += 1
            
        avg_time_per_batch = total_time / processed_batches
        avg_time_per_graph = avg_time_per_batch / batch_size
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"Success!")
        print(f"Avg Time/Batch: {avg_time_per_batch:.2f}s")
        print(f"Avg Time/Graph: {avg_time_per_graph:.4f}s")
        print(f"Peak VRAM Usage: {peak_mem_gb:.2f} GB")
        
        return avg_time_per_graph
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("FAILED: Out of Memory (OOM)")
            torch.cuda.empty_cache()
            return float('inf')
        else:
            raise e

def main():
    sizes = [4, 8, 16, 32, 64, 128]
    results = {}
    for bs in sizes:
        res = test_batch_size(bs)
        results[bs] = res
        if res == float('inf'):
            print("Stopping search due to OOM.")
            break
            
    print("\n\n=== Optimal Batch Size Results ===")
    print("Batch Size | Avg Time/Graph (seconds)")
    for bs, t in results.items():
        time_str = f"{t:.4f}s" if t != float('inf') else "OOM"
        print(f"{bs:10d} | {time_str}")

if __name__ == '__main__':
    main()
