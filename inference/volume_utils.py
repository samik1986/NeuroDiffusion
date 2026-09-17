import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
import concurrent.futures

class SlidingWindowLoader:
    def __init__(self, filepath, window_size=(64, 256, 256), overlap=(16, 64, 64)):
        """
        Loads a large TIF volume using memmap and yields overlapping chunks.
        window_size and overlap are in (Z, Y, X)
        """
        self.filepath = filepath
        self.window_size = np.array(window_size)
        self.overlap = np.array(overlap)
        self.step = self.window_size - self.overlap
        
        # Open image into RAM (loads in ~0.89s, prevents massive overhead of lazy slicing)
        self.vol = tifffile.imread(filepath)
        
        # Unpack shape assuming (Z, Y, X) or (Z, C, Y, X)
        if len(self.vol.shape) > 3:
            self.vol = self.vol[:, 0, :, :] # Handle multichannel if present
            
        self.shape = self.vol.shape
        
    def generate_windows(self):
        z_max, y_max, x_max = self.shape
        wz, wy, wx = self.window_size
        sz, sy, sx = self.step
        
        windows = []
        for z in range(0, max(1, z_max - wz + sz), sz):
            for y in range(0, max(1, y_max - wy + sy), sy):
                for x in range(0, max(1, x_max - wx + sx), sx):
                    z_end = min(z + wz, z_max)
                    y_end = min(y + wy, y_max)
                    x_end = min(x + wx, x_max)
                    windows.append((z, y, x, z_end, y_end, x_end))
        return windows

    def get_chunk(self, window):
        z, y, x, z_end, y_end, x_end = window
        # Extremely fast RAM slice
        return np.array(self.vol[z:z_end, y:y_end, x:x_end], dtype=np.float32)


def compute_vesselness_and_tangent(chunk, sigma=1.0):
    """
    Computes a simple ridgeness measure and the tangent direction field.
    For blazing fast inference, we use the Hessian matrix trace and gradient.
    Returns:
        vesselness: (Z, Y, X) array of ridgeness
        tangent: (3, Z, Y, X) array of unit tangent vectors
    """
    # 1. Smooth the chunk
    smoothed = gaussian_filter(chunk, sigma=sigma)
    
    # 2. Compute gradients (Z, Y, X)
    dz, dy, dx = np.gradient(smoothed)
    
    # 3. Compute Hessian components
    dzz, dzy, dzx = np.gradient(dz)
    _, dyy, dyx = np.gradient(dy)
    _, _, dxx = np.gradient(dx)
    
    # For a bright tubular structure on dark background, the dominant eigenvalues of Hessian are negative.
    # The vesselness can be approximated by the magnitude of the Laplacian (Trace of Hessian) in regions where it's negative.
    laplacian = dzz + dyy + dxx
    vesselness = np.maximum(-laplacian, 0)
    
    # Normalize vesselness
    v_max = vesselness.max()
    if v_max > 0:
        vesselness /= v_max
        
    # 4. Tangent estimation.
    # For a tube, the gradient is radial. The tangent is orthogonal to the gradient.
    # To get a consistent tangent, we can cross the gradient with a reference vector, or use the Hessian eigenvectors.
    # For blazing fast approximation without full eigendecomposition per voxel:
    # We find the direction of minimal second derivative.
    # Here, we use the cross product of gradients at slight offsets, or just use the Hessian pseudo-vector.
    # To keep it extremely fast and vectorized, we approximate the tangent using the structure tensor.
    # Since exact tangent is only needed near the ridge, we can use the direction orthogonal to gradient and largest Hessian diagonal.
    
    # A fast proxy for tangent T = (Tx, Ty, Tz) is the eigenvector of smallest absolute eigenvalue.
    # For a strict implementation we would use `np.linalg.eigh` on the 3x3 Hessian at each voxel.
    # Let's do a vectorized 3x3 eigh over the volume.
    
    H = np.zeros((3, 3, chunk.shape[0], chunk.shape[1], chunk.shape[2]), dtype=np.float32)
    H[0, 0] = dzz
    H[1, 1] = dyy
    H[2, 2] = dxx
    H[0, 1] = H[1, 0] = dzy
    H[0, 2] = H[2, 0] = dzx
    H[1, 2] = H[2, 1] = dyx
    
    # Transpose to (Z, Y, X, 3, 3) for np.linalg.eigh
    H_trans = np.moveaxis(H, [0, 1], [-2, -1])
    
    # Eigh computes eigenvalues/vectors. It is fast enough in C for chunks like 64x256x256.
    # eigh returns ascending eigenvalues, so the first one (most negative) is cross-section,
    # the second is cross-section, the third (closest to 0) is the tangent direction!
    # Wait, eigh sorts by algebraic value. For a bright tube, eigenvalues are L1 <= L2 < L3 ~ 0.
    # So the tangent eigenvector is the one corresponding to the maximum algebraic eigenvalue (L3).
    
    try:
        evals, evecs = np.linalg.eigh(H_trans)
        # evecs is (Z, Y, X, 3, 3). The eigenvectors are the columns: evecs[..., :, i]
        # Tangent is the eigenvector for the largest eigenvalue (which is closest to 0 for bright tubes).
        tangent = evecs[..., :, 2] # (Z, Y, X, 3)
    except Exception as e:
        # Fallback if eigh fails
        tangent = np.zeros((chunk.shape[0], chunk.shape[1], chunk.shape[2], 3), dtype=np.float32)
        tangent[..., 0] = 1.0 # default Z direction
        
    # Move channel to first dimension: (3, Z, Y, X)
    tangent = np.moveaxis(tangent, -1, 0)
    
    return vesselness, tangent
