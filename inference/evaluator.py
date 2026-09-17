import torch
import numpy as np

class ZeroShotEvaluator:
    def __init__(self, heuristic_model, device):
        self.heuristic_model = heuristic_model
        self.device = device
        
    def evaluate_path_batch(self, t1_contexts, t2_contexts, gen_coords, gen_types, 
                            paths_voxels, vesselness_vol, tangent_vol, window_offset):
        N = len(paths_voxels)
        if N == 0:
            return []
            
        t_eval = torch.zeros(N, dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_types_expanded = gen_types.unsqueeze(-1).float()
            predicted_gaps = self.heuristic_model(t1_contexts, t2_contexts, gen_coords, gen_types_expanded, t_eval)
            cost_a_batch = predicted_gaps.cpu().numpy().flatten()
            
        costs = []
        alpha = 1.0
        beta = 1.0 
        for i in range(N):
            cost_b = self._compute_intensity_cost(paths_voxels[i], vesselness_vol, tangent_vol, window_offset)
            costs.append(alpha * cost_a_batch[i] + beta * cost_b)
            
        return costs

    def _compute_intensity_cost(self, path_voxels, vesselness_vol, tangent_vol, window_offset):
        """
        Computes the path integral over the vesselness volume.
        Penalizes misalignment between path direction and volume tangent.
        """
        L = len(path_voxels)
        if L < 2:
            return 0.0
            
        z0, y0, x0 = window_offset
        v_shape = vesselness_vol.shape
        
        total_reward = 0.0
        
        for i in range(L - 1):
            p1 = path_voxels[i]
            p2 = path_voxels[i+1]
            
            # Step vector
            step = p2 - p1
            step_len = np.linalg.norm(step)
            if step_len < 1e-5:
                continue
            step_dir = step / step_len
            
            # Local coordinates in chunk
            lz = int(np.clip(np.round(p1[0] - z0), 0, v_shape[0]-1))
            ly = int(np.clip(np.round(p1[1] - y0), 0, v_shape[1]-1))
            lx = int(np.clip(np.round(p1[2] - x0), 0, v_shape[2]-1))
            
            # Sample vesselness (0 to 1)
            v_val = vesselness_vol[lz, ly, lx]
            
            # Sample image tangent
            t_val = tangent_vol[:, lz, ly, lx]
            t_norm = np.linalg.norm(t_val)
            if t_norm > 1e-5:
                t_val = t_val / t_norm
            else:
                t_val = np.zeros(3)
                
            # Alignment (absolute dot product because tangents don't have inherent orientation)
            alignment = np.abs(np.dot(step_dir, t_val))
            
            # Reward is proportional to vesselness and alignment
            # We want to minimize cost, so reward is negative cost.
            # Base penalty for distance, minus reward for tracking ridgeline
            reward = v_val * alignment * step_len
            total_reward += reward
            
        # Cost is path length minus reward (can be negative if reward is very high)
        # Or simpler: negative average reward per unit length
        path_length = np.sum(np.linalg.norm(np.diff(path_voxels, axis=0), axis=1))
        if path_length > 0:
            avg_reward = total_reward / path_length
            return -avg_reward * 10.0 # scale factor to match Cost A magnitude
        return 0.0
