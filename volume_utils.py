import tifffile
import numpy as np
import scipy.ndimage

def load_volume(tiff_path):
    """
    Loads a multi-channel TIFF volume into a NumPy array.
    Returns array of shape (Z, Y, X) or (C, Z, Y, X).
    """
    print(f"Loading volume from {tiff_path}...")
    volume = tifffile.imread(tiff_path)
    print(f"Volume loaded with shape {volume.shape} and dtype {volume.dtype}")
    
    # If the volume has multiple channels (e.g. 4D shape like C, Z, Y, X),
    # we usually want to extract the structural channel. 
    # For simplicity, if it's 4D, we'll take the first channel.
    if len(volume.shape) == 4:
        # User specified "ch03" in filename, we might need to select the right channel
        # Assuming channel dimension is first or last. Usually Z, Y, X, C or C, Z, Y, X
        # We will assume Z, Y, X for 3D processing. If it's 4D, we just take the max projection or a specific channel.
        # Let's assume C, Z, Y, X for now and take channel 0.
        if volume.shape[0] <= 4: # likely C is first
            volume = volume[0]
        else: # likely C is last
            volume = volume[..., 0]
            
    return volume

def evaluate_ridgeline_intensity(volume, path_coords_microns, voxel_resolution, percentile_threshold=80):
    """
    Interpolates the raw volume intensity along the generated path coordinates.
    Accepts the path if the mean intensity along the path is greater than the specified percentile
    of the local bounding box volume.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        path_coords_microns: list or array of (X, Y, Z) coordinates in microns
        voxel_resolution: (res_x, res_y, res_z) in microns
        percentile_threshold: scalar (0-100), the percentile of local intensity to clear.
        
    Returns:
        bool: True if accepted, False otherwise
        mean_intensity: The calculated mean intensity of the path
    """
    if len(path_coords_microns) == 0:
        return False, 0.0
        
    res_x, res_y, res_z = voxel_resolution
    path_coords = np.array(path_coords_microns)
    
    # Convert microns to voxel indices (X, Y, Z)
    # Note: Images are indexed as (Z, Y, X)
    vox_x = path_coords[:, 0] / res_x
    vox_y = path_coords[:, 1] / res_y
    vox_z = path_coords[:, 2] / res_z
    
    # Ensure coordinates are within volume bounds
    Z_max, Y_max, X_max = volume.shape
    vox_x = np.clip(vox_x, 0, X_max - 1)
    vox_y = np.clip(vox_y, 0, Y_max - 1)
    vox_z = np.clip(vox_z, 0, Z_max - 1)
    
    # Coordinates for scipy map_coordinates must be (Z, Y, X) to match volume shape
    coords_zyx = np.vstack([vox_z, vox_y, vox_x])
    
    # Interpolate intensities along the path
    path_intensities = scipy.ndimage.map_coordinates(volume, coords_zyx, order=1, mode='nearest')
    mean_path_intensity = np.mean(path_intensities)
    
    # Calculate local volume threshold
    # Extract a bounding box around the path to find the local ridgeline percentile
    pad = 10 # 10 voxels padding
    min_z, max_z = int(np.min(vox_z)), int(np.max(vox_z))
    min_y, max_y = int(np.min(vox_y)), int(np.max(vox_y))
    min_x, max_x = int(np.min(vox_x)), int(np.max(vox_x))
    
    min_z, max_z = max(0, min_z - pad), min(Z_max, max_z + pad + 1)
    min_y, max_y = max(0, min_y - pad), min(Y_max, max_y + pad + 1)
    min_x, max_x = max(0, min_x - pad), min(X_max, max_x + pad + 1)
    
    local_vol = volume[min_z:max_z, min_y:max_y, min_x:max_x]
    
    if local_vol.size == 0:
        return False, mean_path_intensity
        
    threshold_val = np.percentile(local_vol, percentile_threshold)
    
    is_accepted = mean_path_intensity > threshold_val
    return is_accepted, mean_path_intensity
