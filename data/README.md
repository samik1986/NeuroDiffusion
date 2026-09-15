# Data Processing

This module handles the loading, parsing, and formatting of raw neuronal morphology data (SWC files) for use in the NeuroDiffusion pipeline.

## Files

### 1. `data_loader.py`
**Role:** Main PyTorch `Dataset` and `DataLoader` implementation.
**Key Features:**
- **`SWCParser`:** Parses `.swc` files, which describe the 3D topology of a neuron (ID, Type, X, Y, Z, Radius, Parent ID).
- **Coordinate Centering:** Implements spatial translation invariance by mean-centering the graph coordinates (`centered_coords = coords - centroid`).
- **Path Extraction for Sequence Generation:** Extracts a ground-truth random walk path (`path_coords`, `path_types`) from the tree to train the `SequenceGenerator`.
- **Normalization:** Normalizes branch node coordinates based on the bounding box to ensure the standard DDPM Gaussian noise scale operates correctly.
- **Sparse Eigendecomposition Optimization:** Removed costly `eigsh` Laplacian Positional Encoding computation from the CPU loader loop to eliminate CPU multiprocessing bottlenecks and avoid target leakage in the Backward Diffusion model.

### 2. `preprocess_dataset.py`
**Role:** Script to preemptively filter and prepare the raw SWC dataset.
**Usage:** Run this independently if you need to pre-cache or clean a raw dataset before passing it into `train.py`.

## Assumptions
- **Soma Ignored:** By default (`ignore_soma: true` in `config.yaml`), the root soma node (type 1) is ignored during topological graph extraction.
- **Max Nodes:** Trees are truncated or split based on `max_nodes` to ensure they fit in GPU memory and can be batched efficiently.
- **Undirected Graphs:** Topologies are loaded as undirected graphs (both `parent->child` and `child->parent` edges are added) so that standard Graph Neural Networks can propagate messages bidirectionally.
