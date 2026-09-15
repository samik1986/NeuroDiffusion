import os
import torch
import numpy as np
import networkx as nx

from utils.utils import load_config
from utils.tree_utils import parse_swc_to_graph, separate_trees, find_nearest_neighbors
from utils.volume_utils import load_volume, evaluate_ridgeline_intensity
from models.backward_diffusion import BackwardDiffusionModel
from models.sequence_generator import SequenceGenerator
from models.heuristic_evaluator import ValidityHeuristicEvaluator

def run_inference():
    print("Starting NeuroDiffusion Inference Pipeline...")
    
    # 1. Load Configurations
    config = load_config('config.yaml')
    inf_params = config.get('inference', {})
    
    swc_path = inf_params.get('input_swc_path')
    tiff_path = inf_params.get('raw_volume_path')
    voxel_resolution = inf_params.get('voxel_resolution', [0.1102, 0.1102, 0.5])
    max_distance = inf_params.get('max_joining_distance', 10.0)
    ridgeline_threshold = inf_params.get('ridgeline_percentile_threshold', 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load Models
    print("Loading Generative and Heuristic Models...")
    # In a real scenario, you would load state_dicts here (e.g., model.load_state_dict(torch.load('weights.pth')))
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
    
    print(f"Loading RAW Volume: {tiff_path}")
    if not os.path.exists(tiff_path):
        print(f"TIFF file not found: {tiff_path}. Volume thresholding will be skipped or aborted.")
        return
        
    volume = load_volume(tiff_path)
    
    # 4. Find Nearest Neighbor Candidates
    print(f"Searching for candidate connections within {max_distance} microns...")
    candidates = find_nearest_neighbors(trees, max_distance=max_distance)
    print(f"Found {len(candidates)} candidate connections between trees.")
    
    # 5. Evaluate and Join
    joined_edges = []
    
    for idx, (tree_i_idx, n_i, tree_j_idx, n_j, dist) in enumerate(candidates):
        print(f"\nEvaluating Candidate {idx+1}/{len(candidates)}: Tree {tree_i_idx} (Node {n_i}) -> Tree {tree_j_idx} (Node {n_j}) | Dist: {dist:.2f}um")
        
        # --- Simulate extracting context embeddings for the trees ---
        # In practice, these would be the actual features passed through the graph encoder
        t1_context = torch.rand(1, 128, device=device) 
        t2_context = torch.rand(1, 128, device=device)
        
        with torch.no_grad():
            # A. Autoregressively draw sequence path from n_i
            # Provide the starting coordinate in actual space
            start_coord = torch.tensor(swc_graph.nodes[n_i]['pos'], dtype=torch.float32, device=device).unsqueeze(0)
            
            # Draw sample sequence
            gen_coords, gen_types = seq_generator.sample(t1_context, start_coord)
            
            # B. Neural Heuristic Evaluator
            # Checks if the drawn sequence makes structural sense connecting T1 and T2
            validity_logit = heuristic_eval(t1_context, t2_context, gen_coords, gen_types.unsqueeze(-1).float())
            validity_prob = torch.sigmoid(validity_logit).item()
            
            print(f"  Neural Heuristic Probability: {validity_prob:.4f}")
            
            if validity_prob > 0.5:
                # C. Volume Ridgeline Evaluation
                # Convert generated path tensor to numpy list
                path_microns = gen_coords[0].cpu().numpy()
                
                is_accepted, mean_intensity = evaluate_ridgeline_intensity(
                    volume, 
                    path_microns, 
                    voxel_resolution, 
                    percentile_threshold=ridgeline_threshold
                )
                
                print(f"  Volume Ridgeline Mean Intensity: {mean_intensity:.2f} (Accepted: {is_accepted})")
                
                if is_accepted:
                    print("  [SUCCESS] Connection Accepted! Joining trees.")
                    joined_edges.append((n_i, n_j, path_microns))
                else:
                    print("  [REJECTED] Path does not align with raw intensity ridgeline.")
            else:
                print("  [REJECTED] Neural Heuristic deemed connection invalid.")
                
    # 6. Final Output Generation
    print(f"\nInference Complete. Successfully established {len(joined_edges)} new sub-neuronal connections.")
    # Here we would export the joined_edges and trees back into a unified SWC file format.
    # e.g., export_swc(swc_graph, joined_edges, 'output_reconstructed.swc')

if __name__ == '__main__':
    run_inference()
