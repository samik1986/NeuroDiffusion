import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
import multiprocessing as mp
from tqdm import tqdm
from data.data_loader import SWCParser, NeuroDiffusionDataset
import collections
import random

DATA_DIR = "/mnt/diskg9-3/NeuroGramLM/SWCs"
OUT_DIR = "/mnt/diskg9-3/NeuroGramLM/processed_chunks"
CHUNK_SIZE = 1000

# Helper function to process a single file and generate ALL crops
def process_file(swc_path):
    parser = SWCParser(swc_path)
    nodes, edges = parser.parse(ignore_soma=True)
    if len(nodes) == 0:
        return []
        
    dataset = NeuroDiffusionDataset(DATA_DIR) # dummy to use _extract_features
    MAX_NODES = dataset.max_nodes
    OVERLAP = dataset.overlap
    
    if len(nodes) <= MAX_NODES:
        return [dataset._extract_features(nodes, edges)]
        
    adj = collections.defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    uncovered = set(nodes.keys())
    crops = []
    
    while uncovered:
        start_node = random.choice(list(uncovered))
        visited = set([start_node])
        visited_list = [start_node]
        queue = collections.deque([start_node])
        
        while queue and len(visited) < MAX_NODES:
            curr = queue.popleft()
            neighbors = list(adj[curr])
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    visited_list.append(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= MAX_NODES:
                        break
                        
        crop_nodes = {n: nodes[n] for n in visited}
        crop_edges = [(u, v) for u, v in edges if u in visited and v in visited]
        
        try:
            crops.append(dataset._extract_features(crop_nodes, crop_edges))
        except Exception:
            pass
            
        core_nodes = set(visited_list[:max(1, len(visited_list) - OVERLAP)])
        uncovered = uncovered - core_nodes
        
    return crops

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    swc_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.swc'):
                swc_files.append(os.path.join(root, file))
    print(f"Found {len(swc_files)} files. Preprocessing...")
    
    pool = mp.Pool(processes=32)
    chunk_data = []
    chunk_idx = 0
    
    # Process files with file-level progress bar
    for crops in tqdm(pool.imap_unordered(process_file, swc_files, chunksize=1), total=len(swc_files), desc="Processing SWCs"):
        if crops:
            chunk_data.append(crops)
            
        if len(chunk_data) >= CHUNK_SIZE:
            out_path = os.path.join(OUT_DIR, f"chunk_{chunk_idx}.pt")
            torch.save(chunk_data, out_path)
            # print(f"Saved chunk {chunk_idx} with {len(chunk_data)} SWC items")
            chunk_data = []
            chunk_idx += 1
            
    # Save remaining
    if len(chunk_data) > 0:
        out_path = os.path.join(OUT_DIR, f"chunk_{chunk_idx}.pt")
        torch.save(chunk_data, out_path)
        
    pool.close()
    pool.join()
if __name__ == "__main__":
    main()
