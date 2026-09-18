import os
import torch
import numpy as np
import networkx as nx
from collections import defaultdict

from utils.utils import load_config
from utils.tree_utils import parse_swc_to_graph, separate_trees, find_nearest_neighbors
from models.backward_diffusion import BackwardDiffusionModel
from models.sequence_generator import SequenceGenerator
from models.heuristic_evaluator import ValidityHeuristicEvaluator

def run_inference():
    print("Starting NeuroDiffusion Inference Pipeline (Top-5 Gap Minimization)...")
    
    # 1. Load Configurations
    config = load_config('config.yaml')
    inf_params = config.get('inference', {})
    
    swc_path = inf_params.get('input_swc_path', '/mnt/diskg9-3/NeuroGramLM/SWCs/sample.swc')
    max_distance = inf_params.get('max_joining_distance', 10.0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load Models
    print("Loading Generative and Heuristic Models...")
    # In a real scenario, you would load state_dicts here
    backward_model = BackwardDiffusionModel(config=config).to(device).eval()
    seq_generator = SequenceGenerator(config=config).to(device).eval()
    heuristic_eval = ValidityHeuristicEvaluator(config=config).to(device).eval()
    
    # 3. Load Data
    print(f"Parsing SWC: {swc_path}")
    if not os.path.exists(swc_path):
        print(f"SWC file not found: {swc_path}. Please place the file in the correct directory.")
        return
        
    swc_graph = parse_swc_to_graph(swc_path)
    trees = separate_trees(swc_graph)
    print(f"Graph separated into {len(trees)} distinct trees.")
    
    # 4. Find Nearest Neighbor Candidates
    print(f"Searching for candidate connections within {max_distance} microns...")
    candidates = find_nearest_neighbors(trees, max_distance=max_distance)
    print(f"Found {len(candidates)} candidate connections between trees.")
    
    # Group candidates by source tree
    tree_candidates = defaultdict(list)
    for c in candidates:
        tree_i_idx, n_i, tree_j_idx, n_j, dist = c
        tree_candidates[tree_i_idx].append(c)
        
    # 5. Evaluate and Join
    joined_edges = []
    t_eval = torch.zeros(1, dtype=torch.long, device=device)
    
    for tree_i_idx, group in tree_candidates.items():
        # Sort by Euclidean distance and take Top-5 candidates
        top5_group = sorted(group, key=lambda x: x[4])[:5]
        
        print(f"\nEvaluating Top-{len(top5_group)} Candidates for Tree {tree_i_idx}")
        
        best_gap = float('inf')
        best_candidate = None
        best_path = None
        
        # Simulated context for Tree 1
        t1_context = torch.rand(1, 128, device=device)
        
        for idx, (ti, n_i, tree_j_idx, n_j, dist) in enumerate(top5_group):
            t2_context = torch.rand(1, 128, device=device)
            
            with torch.no_grad():
                # A. Autoregressively draw sequence path to connect T1 and T2
                gen_coords, gen_types, _ = seq_generator.sample(t1_context, t2_context)
                
                # B. Neural Heuristic Evaluator Predicts the Structural Gap
                # The model was trained to predict the SE(3) scale-invariant MSE gap to the unknown true biological path.
                # Lower predicted gap = better connection.
                predicted_gap = heuristic_eval(t1_context, t2_context, gen_coords, gen_types.unsqueeze(-1).float(), t_eval)
                gap_val = predicted_gap.item()
                
                print(f"  Candidate {idx+1} -> Tree {tree_j_idx} | Dist: {dist:.2f}um | Predicted Structural Gap: {gap_val:.4f}")
                
                if gap_val < best_gap:
                    best_gap = gap_val
                    best_candidate = (ti, n_i, tree_j_idx, n_j)
                    best_path = gen_coords[0].cpu().numpy()
                    
        # C. Select the Best Candidate
        if best_candidate is not None:
            _, src_n, tgt_tree, tgt_n = best_candidate
            print(f"  [SUCCESS] Selected Best Connection to Tree {tgt_tree} (Predicted Gap: {best_gap:.4f})")
            joined_edges.append((src_n, tgt_n, best_path))
                
    # 6. Final Output Generation
    print(f"\nInference Complete. Successfully established {len(joined_edges)} new sub-neuronal connections.")
    # Export code would go here
    # export_swc(swc_graph, joined_edges, 'output_reconstructed.swc')

if __name__ == '__main__':
    run_inference()
