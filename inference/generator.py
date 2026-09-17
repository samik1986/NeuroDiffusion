import torch
import numpy as np

class InferenceGenerator:
    def __init__(self, seq_generator, device):
        self.seq_generator = seq_generator
        self.device = device
        
    def connect_endpoints_batch(self, pairs, t1_contexts, t2_contexts):
        """
        pairs: list of (p1, p2) numpy arrays (shape (N, 2, 3))
        """
        N = len(pairs)
        if N == 0:
            return None, None, []
            
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_coords, gen_types, _ = self.seq_generator.sample(t1_contexts, t2_contexts, num_timesteps=30)
            
        coords = gen_coords.cpu().numpy()
        types = gen_types.cpu().numpy()
        
        mapped_paths = []
        for i in range(N):
            c = coords[i]
            p1, p2 = pairs[i]
            if len(c) >= 2:
                start_c = c[0]
                end_c = c[-1]
                vec_c = end_c - start_c
                vec_p = p2 - p1
                norm_c = np.linalg.norm(vec_c)
                if norm_c > 1e-5:
                    scale = np.linalg.norm(vec_p) / norm_c
                    t = np.linspace(0, 1, len(c))[:, np.newaxis]
                    baseline_p = p1 + t * vec_p
                    baseline_c = start_c + t * vec_c
                    high_freq_c = c - baseline_c
                    mapped_c = baseline_p + high_freq_c * scale
                else:
                    mapped_c = c
                if len(mapped_c) >= 2:
                    mapped_c[0] = p1
                    mapped_c[-1] = p2
            else:
                mapped_c = c
            mapped_paths.append(mapped_c)
            
        return gen_coords, gen_types, mapped_paths

    def extend_endpoint_batch(self, points, tangents, t1_contexts):
        pairs = []
        for i in range(len(points)):
            p2 = points[i] + tangents[i] * 20.0
            pairs.append((points[i], p2))
        return self.connect_endpoints_batch(pairs, t1_contexts, t1_contexts)
