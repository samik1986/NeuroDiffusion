import os
import torch
import numpy as np
import concurrent.futures
import threading
import queue
from tqdm import tqdm
import networkx as nx

from utils.utils import load_config
from models.sequence_generator import SequenceGenerator
from models.heuristic_evaluator import ValidityHeuristicEvaluator

from inference.volume_utils import SlidingWindowLoader, compute_vesselness_and_tangent
from inference.graph_utils import parse_swc_to_graph, extract_endpoints_and_trees, get_endpoints_in_window
from inference.generator import InferenceGenerator
from inference.evaluator import ZeroShotEvaluator

def process_window(window, loader):
    """Producer function: loads chunk and computes vesselness/tangent on CPU."""
    chunk = loader.get_chunk(window)
    # If chunk is mostly zero (background), skip heavy compute
    if chunk.max() < 1e-3:
        return window, None, None, chunk
        
    vesselness, tangent = compute_vesselness_and_tangent(chunk, sigma=1.0)
    return window, vesselness, tangent, chunk

def export_swc(G, new_edges, out_path, voxel_res):
    """
    Exports the combined graph to SWC.
    voxel_res: [X_um, Y_um, Z_um]
    """
    # Create a new graph and map everything back to microns
    scale = np.array([voxel_res[2], voxel_res[1], voxel_res[0]]) # Z, Y, X
    
    max_id = 0
    visited = set()
    with open(out_path, 'w') as f:
        f.write("# Reconstructed SWC\n")
        # Write original nodes
        for node in G.nodes():
            pos = G.nodes[node]['pos'] * scale
            r = G.nodes[node]['radius']
            t = G.nodes[node]['type_id']
            f.write(f"{node} {t} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {r:.4f} -1\n")
            visited.add(node)
            max_id = max(max_id, node)
            
    with open(out_path, 'a') as f:
        # Write new edges
        curr_id = max_id + 1
        for (n1, n2, path, cost) in new_edges:
            prev_id = n1
            for i in range(1, len(path)):
                p = path[i] * scale
                f.write(f"{curr_id} 0 {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 1.0 {prev_id}\n")
                prev_id = curr_id
                curr_id += 1


def gpu_consumer(device_id, task_queue, result_list, list_lock, config):
    """Consumer function: runs models on a dedicated GPU."""
    device = torch.device(f'cuda:{device_id}')
    
    # Load Models onto this specific GPU
    seq_gen = SequenceGenerator(config=config).to(device).eval()
    heuristic_model = ValidityHeuristicEvaluator(config=config).to(device).eval()
    
    ckpt_gen = 'output/checkpoints/gen_epoch_40.pth'
    ckpt_eval = 'output/checkpoints/eval_epoch_40.pth'
    if os.path.exists(ckpt_gen):
        state_dict = torch.load(ckpt_gen, map_location=device, weights_only=True)
        seq_gen.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    if os.path.exists(ckpt_eval):
        state_dict = torch.load(ckpt_eval, map_location=device, weights_only=True)
        heuristic_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        
    # Turbo-charge the models
    try:
        seq_gen = torch.compile(seq_gen)
        heuristic_model = torch.compile(heuristic_model)
    except:
        pass
        
    generator = InferenceGenerator(seq_gen, device)
    evaluator = ZeroShotEvaluator(heuristic_model, device)
    
    while True:
        task = task_queue.get()
        if task is None: # Sentinel value to stop thread
            task_queue.task_done()
            break
            
        window, vesselness, tangent, eps_in_window = task
        window_offset = (window[0], window[1], window[2])
        
        t_context = torch.rand(1, 128, device=device)
        local_connections = []
        
        # Step 7: Connect Subgraphs
        pairs = []
        meta = []
        for i in range(len(eps_in_window)):
            for j in range(i+1, len(eps_in_window)):
                t1_idx, n1, p1 = eps_in_window[i]
                t2_idx, n2, p2 = eps_in_window[j]
                
                if t1_idx == t2_idx: continue
                if np.linalg.norm(p1 - p2) > 100.0: continue
                
                pairs.append((p1, p2))
                meta.append((n1, n2))
                
        if pairs:
            N = len(pairs)
            chunk_size = 1024 # Massive Batch Size for GPU
            for chunk_idx in range(0, N, chunk_size):
                end_idx = min(chunk_idx + chunk_size, N)
                c_pairs = pairs[chunk_idx:end_idx]
                c_meta = meta[chunk_idx:end_idx]
                c_N = len(c_pairs)
                
                c_t1 = torch.rand(c_N, 128, device=device)
                c_t2 = torch.rand(c_N, 128, device=device)
                
                gen_c, gen_t, mapped_paths = generator.connect_endpoints_batch(c_pairs, c_t1, c_t2)
                if mapped_paths:
                    costs = evaluator.evaluate_path_batch(c_t1, c_t2, gen_c, gen_t, mapped_paths, vesselness, tangent, window_offset)
                    
                    for k in range(c_N):
                        if costs[k] < 0:
                            local_connections.append((c_meta[k][0], c_meta[k][1], mapped_paths[k], costs[k]))
                            
        # Step 8: Extend isolated endpoints
        ext_points = []
        ext_tangents = []
        ext_meta = []
        for i in range(len(eps_in_window)):
            t_idx, n, p = eps_in_window[i]
            z0, y0, x0 = window_offset
            lz = int(np.clip(np.round(p[0] - z0), 0, vesselness.shape[0]-1))
            ly = int(np.clip(np.round(p[1] - y0), 0, vesselness.shape[1]-1))
            lx = int(np.clip(np.round(p[2] - x0), 0, vesselness.shape[2]-1))
            
            t_dir = tangent[:, lz, ly, lx]
            t_norm = np.linalg.norm(t_dir)
            if t_norm > 1e-5:
                t_dir = t_dir / t_norm
                ext_points.append(p)
                ext_tangents.append(t_dir)
                ext_meta.append(n)
                
        if ext_points:
            N_ext = len(ext_points)
            chunk_size = 1024
            for chunk_idx in range(0, N_ext, chunk_size):
                end_idx = min(chunk_idx + chunk_size, N_ext)
                c_pts = ext_points[chunk_idx:end_idx]
                c_tng = ext_tangents[chunk_idx:end_idx]
                c_meta = ext_meta[chunk_idx:end_idx]
                c_N = len(c_pts)
                
                c_t1 = torch.rand(c_N, 128, device=device)
                
                gen_c, gen_t, mapped_paths = generator.extend_endpoint_batch(c_pts, c_tng, c_t1)
                if mapped_paths:
                    costs = evaluator.evaluate_path_batch(c_t1, c_t1, gen_c, gen_t, mapped_paths, vesselness, tangent, window_offset)
                    
                    for k in range(c_N):
                        if costs[k] < -1.0:
                            local_connections.append((c_meta[k], None, mapped_paths[k], costs[k]))
                    
        # Thread-safe append
        if local_connections:
            with list_lock:
                result_list.extend(local_connections)
                
        task_queue.task_done()


def run_pipeline():
    print("Starting Blazing Fast NeuroDiffusion Inference...")
    
    # Enable CPU/GPU Speedups
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    config = load_config('config.yaml')
    inf_params = config.get('inference', {})
    
    raw_vol_path = inf_params.get('raw_volume_path', '/mnt/diskg9-3/NeuroGramLM/F0046_multichannel_cmle_ch03.tif')
    swc_path = inf_params.get('input_swc_path', '/mnt/diskg9-3/NeuroGramLM/skeletons_connected_new.swc')
    voxel_res = inf_params.get('voxel_resolution', [0.1102, 0.1102, 0.5]) # X, Y, Z
    
    scale_um_to_vox = np.array([voxel_res[2], voxel_res[1], voxel_res[0]])
    
    print(f"Parsing SWC: {swc_path}")
    G = parse_swc_to_graph(swc_path)
    
    # Scale all positions to voxel space
    for node in G.nodes():
        G.nodes[node]['pos'] = G.nodes[node]['pos'] / scale_um_to_vox
        
    trees, endpoints = extract_endpoints_and_trees(G)
    print(f"Extracted {len(trees)} subgraphs with {len(endpoints)} endpoints.")
    
    print("Initializing Sliding Window Loader...")
    window_size = (64, 256, 256)
    overlap = (16, 64, 64)
    loader = SlidingWindowLoader(raw_vol_path, window_size=window_size, overlap=overlap)
    windows = loader.generate_windows()
    print(f"Total Windows to process: {len(windows)}")
    
    # Concurrency Architecture: CPU Producer -> Queue -> Multi-GPU Consumers
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No GPUs found. Running consumer on CPU.")
        num_gpus = 1
        
    print(f"Spinning up {num_gpus} GPU Consumer Threads...")
    
    # CPU Queue with maxsize buffers the next set of data while GPU computes
    task_queue = queue.Queue(maxsize=16) 
    new_connections = []
    list_lock = threading.Lock()
    
    gpu_threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=gpu_consumer, args=(i, task_queue, new_connections, list_lock, config))
        t.start()
        gpu_threads.append(t)
    
    print("Starting CPU Processing Pool for Volumetric Analysis...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_window = {executor.submit(process_window, w, loader): w for w in windows}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_window), total=len(windows)):
            window, vesselness, tangent, chunk = future.result()
            if vesselness is None:
                continue
                
            eps_in_window = get_endpoints_in_window(endpoints, window, margin=0)
            if len(eps_in_window) < 1:
                continue
                
            # Block and wait if the GPU queue is full (CPU Buffering)
            task_queue.put((window, vesselness, tangent, eps_in_window))
            
    # Send sentinels to stop GPU threads
    for _ in range(num_gpus):
        task_queue.put(None)
        
    for t in gpu_threads:
        t.join()
                        
    print(f"Generated {len(new_connections)} valid topological additions.")
    
    out_swc_path = 'output/test/reconstructed.swc'
    export_swc(G, new_connections, out_swc_path, voxel_res)
    print(f"Exported Reconstructed SWC to {out_swc_path}")

if __name__ == "__main__":
    run_pipeline()
