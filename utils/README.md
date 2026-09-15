# Utilities

This module contains various helper functions, math operations, and loss logic used throughout the NeuroDiffusion pipeline. 

## Files

### 1. `loss_functions.py`
**Role:** Centralizes the loss computations for the models.
**Features:**
- `edge_loss_fn`: Computes Binary Cross-Entropy (BCE) with Logits for the Backward Diffusion model. Learns to classify which of the candidate topological edges are true connections vs. randomly sampled negative connections.
- `coord_loss_fn`: Computes Mean Squared Error (MSE) for the Sequence Generator to regress the 3D continuous branch coordinates against the added DDPM Gaussian noise.
- `type_loss_fn`: Computes standard Cross-Entropy for the categorical prediction of node branch types.

### 2. `tree_utils.py`
**Role:** Algorithms for parsing and manipulating neuronal tree structures.
**Features:**
- Converts raw SWC dictionaries into `networkx` graphs.
- Discovers independent sub-trees and connected components.
- Locates nearest spatial neighbors in 3D space to assist in heuristics evaluation and loss calculation.

### 3. `volume_utils.py`
**Role:** Algorithms for interacting with raw 3D biological volume data (e.g., TIFF stacks).
**Features:**
- Loading multi-channel voxel arrays.
- Implements `evaluate_ridgeline_intensity`, an algorithm that traverses the generated 3D branch paths and queries the underlying raw biological volume data to verify if the path correctly follows high-intensity ridges (actual physical neurites).

### 4. `utils.py`
**Role:** Generic OS and configuration helpers.
**Features:**
- Safely parses and loads the YAML hyperparameters from `config.yaml`.
