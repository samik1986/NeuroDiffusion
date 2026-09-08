import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SWCParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.nodes = {}  # id -> {type, x, y, z, radius, parent}
        self.edges = []  # list of (parent_id, child_id)
        
    def parse(self, ignore_soma=True):
        """Parses the SWC file, optionally ignoring soma (type 1) nodes."""
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 7:
                    continue
                
                n_id = int(parts[0])
                n_type = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                radius = float(parts[5])
                parent_id = int(parts[6])
                
                if ignore_soma and n_type == 1:
                    continue
                    
                self.nodes[n_id] = {
                    'type': n_type,
                    'coord': np.array([x, y, z]),
                    'radius': radius,
                    'parent': parent_id
                }
                
                if parent_id != -1 and parent_id in self.nodes:
                    self.edges.append((parent_id, n_id))
                    
        return self.nodes, self.edges

class NeuroDiffusionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.swc_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.swc'):
                    self.swc_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.swc_files)} SWC files in {data_dir}...")
                
    def _process_graph(self, nodes, edges):
        # 1. Translation Invariance: Center coordinates to centroid
        coords = np.array([node['coord'] for node in nodes.values()])
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        
        node_ids = list(nodes.keys())
        id_to_idx = {n_id: i for i, n_id in enumerate(node_ids)}
        
        # 2. Scale Invariance: Normalize by max bounding box dimension
        max_dim = np.max(np.abs(centered_coords))
        if max_dim > 0:
            normalized_coords = centered_coords / max_dim
        else:
            normalized_coords = centered_coords
            
        # 3. Rotation Invariance Features (distance to parent, angle with grandparent segment)
        features = []
        for i, n_id in enumerate(node_ids):
            node = nodes[n_id]
            parent_id = node['parent']
            
            dist = 0.0
            angle = 0.0
            
            if parent_id in nodes:
                parent_coord = nodes[parent_id]['coord']
                vec_p_c = node['coord'] - parent_coord
                dist = np.linalg.norm(vec_p_c)
                
                grandparent_id = nodes[parent_id]['parent']
                if grandparent_id in nodes:
                    gparent_coord = nodes[grandparent_id]['coord']
                    vec_gp_p = parent_coord - gparent_coord
                    
                    norm_p_c = np.linalg.norm(vec_p_c)
                    norm_gp_p = np.linalg.norm(vec_gp_p)
                    
                    if norm_p_c > 0 and norm_gp_p > 0:
                        cos_theta = np.dot(vec_p_c, vec_gp_p) / (norm_p_c * norm_gp_p)
                        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
            features.append([dist, angle])
            
        # Convert to tensors
        x = torch.tensor(normalized_coords, dtype=torch.float32)
        inv_features = torch.tensor(features, dtype=torch.float32)
        
        edge_index = []
        for p, c in edges:
            if p in id_to_idx and c in id_to_idx:
                edge_index.append([id_to_idx[p], id_to_idx[c]])
                edge_index.append([id_to_idx[c], id_to_idx[p]]) # Undirected graph for standard GNN processing
                
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return {
            'x': x,
            'inv_features': inv_features,
            'edge_index': edge_index,
            'centroid': torch.tensor(centroid, dtype=torch.float32),
            'max_dim': float(max_dim)
        }

    def __len__(self):
        return len(self.swc_files)

    def __getitem__(self, idx):
        file = self.swc_files[idx]
        parser = SWCParser(file)
        nodes, edges = parser.parse(ignore_soma=True)
        if len(nodes) == 0:
            # If the file is empty or only had somas, fallback to another index
            # This is a simple fallback, for a robust dataloader we would pre-filter empty ones
            return self.__getitem__((idx + 1) % len(self))
            
        return self._process_graph(nodes, edges)

def custom_collate_fn(batch):
    """
    Batches multiple graphs into a single large disconnected graph (standard GNN batching).
    """
    # Filter out None values (failed parsing)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    x_list = []
    inv_features_list = []
    edge_index_list = []
    batch_idx_list = []
    
    node_offset = 0
    for i, graph in enumerate(batch):
        num_nodes = graph['x'].shape[0]
        
        x_list.append(graph['x'])
        inv_features_list.append(graph['inv_features'])
        
        # Offset edge indices by the current number of nodes
        if graph['edge_index'].shape[1] > 0:
            edge_index_list.append(graph['edge_index'] + node_offset)
            
        # Create a batch tensor to keep track of which graph each node belongs to
        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        node_offset += num_nodes
        
    x_batch = torch.cat(x_list, dim=0)
    inv_features_batch = torch.cat(inv_features_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    
    if len(edge_index_list) > 0:
        edge_index_batch = torch.cat(edge_index_list, dim=1)
    else:
        edge_index_batch = torch.empty((2, 0), dtype=torch.long)
        
    return {
        'x': x_batch,
        'inv_features': inv_features_batch,
        'edge_index': edge_index_batch,
        'batch': batch_idx
    }

def get_dataloader(config):
    """
    Returns a train_loader and val_loader based on the config.
    """
    dl_config = config.get('dataloader', {})
    batch_size = dl_config.get('batch_size', 2)
    num_workers = dl_config.get('num_workers', 4)
    train_split = dl_config.get('train_split', 0.8)
    
    dataset = NeuroDiffusionDataset(config['data']['swc_dir'])
    
    total_len = len(dataset)
    if total_len == 0:
        # Fallback if no files found
        return None, None
        
    train_len = int(total_len * train_split)
    val_len = total_len - train_len
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    from utils import load_config
    
    # Test the parallel data loader with config.yaml
    try:
        config = load_config("config.yaml")
        data_path = config['data']['swc_dir']
        
        if os.path.exists(data_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device} for optimization")
            
            dataloader = get_dataloader(config)
            
            for batch in dataloader:
                if batch is None: continue
                
                # Parallelization: Data loaded on CPU via num_workers, transferred to GPU seamlessly
                x = batch['x'].to(device, non_blocking=True)
                edge_index = batch['edge_index'].to(device, non_blocking=True)
                
                print(f"Batched Graph on {device}: {x.shape[0]} nodes, {edge_index.shape[1]//2} edges (from {config['dataloader']['batch_size']} files)")
                break
        else:
            print(f"Directory {data_path} not found. Please verify the path in config.yaml.")
    except Exception as e:
        print(f"Error loading config: {e}")
