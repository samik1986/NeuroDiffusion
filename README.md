# NeuroDiffusion

NeuroDiffusion is a generative machine learning pipeline designed to reconstruct fragmented 3D neuronal topologies from SWC data. By framing the problem as a reversal of a fragmentation (diffusion) process, the system learns to probabilistically infer missing topological connections and synthesize the 3D biological coordinates bridging them.

## Documentation Index

The NeuroDiffusion project is comprehensively documented to explain both the mathematical theory and the software engineering architecture. Please refer to the following documents for deep dives:

**Core Theory & Architecture:**
- [Algorithm Formulation & Tuning Guide (`algo.md`)](./algo.md)
- [High-Level Architecture (`architecture.md`)](./architecture.md)
- [Layer-wise Unrolled Architecture (`model_arch.md`)](./model_arch.md)
- [Loss Functions Formulation (`losses.md`)](./losses.md)
- [Invariance Properties & Theory (`invariance.md`)](./invariance.md)

**Module-Specific Documentation:**
- [Models (`models/README.md`)](./models/README.md)
- [Trainers (`trainers/README.md`)](./trainers/README.md)
- [Inference Pipeline (`inference/README.md`)](./inference/README.md)
- [Data Processing (`data/README.md`)](./data/README.md)
- [Utilities (`utils/README.md`)](./utils/README.md)
## Repository Structure

The codebase is organized into modular directories for maintainability. Please see the individual `README.md` in each folder for deep dives into their specific architectures:

- `models/`: The core neural networks, including the hybrid EGNN-Transformer **Backward Diffusion Model**, the EGNN-based **Sequence Generator**, the **Validity Heuristic Evaluator**, and the **Discrete Forward Diffusion** noise engine.
- `trainers/`: Isolated PyTorch optimization loops utilizing Automatic Mixed Precision (AMP) and DDP.
- `data/`: Custom PyTorch DataLoaders, coordinate normalization, and dataset preprocessing scripts for raw `.swc` files.
- `utils/`: Custom BCE/MSE loss formulations, `networkx` topological tree manipulations, and biological 3D volume (TIFF) ridgeline intensity evaluations.
- `scripts/`: Development utility scripts, profiling, and batch-size tuners.

## Algorithm & Architecture Overview

### 1. Forward Diffusion (Fragmentation)
Instead of standard continuous coordinate noise, we apply a **discrete diffusion process** over the graph topology $G = (V, E)$. At $t=0$, the tree is fully connected. As $t \to 100$, edges are incrementally masked out, fracturing the single neuron into isolated, disconnected branches. 

### 2. Backward Diffusion (Topological Reconstruction)
The **Backward Diffusion Model** learns to reverse this discrete fragmentation. Given the scattered fragments (represented as the surviving `noisy_edge_index`), a custom Graph Convolutional Network (using `scatter_add` for dynamic topology processing) exchanges spatial messages locally. A global `TransformerEncoder` then aggregates cross-fragment information. Finally, the model outputs a predicted binary linkage probability for a set of candidate edges to "stitch" the fragments back together.
**Assumption**: Target leakage is heavily guarded against; ground-truth Laplacians (`lap_pe`) are explicitly excluded so the model must learn *geometric* structural rules rather than memorizing eigendecompositions.

### 3. Sequence Generation (Coordinate Regression)
Once the Backward model predicts *that* two fragments should connect, the **Sequence Generator** predicts *how*. A continuous DDPM is used to synthesize a smooth 3D trajectory connecting the fragments. Starting from pure $N(0, I)$ Gaussian noise, a deep 1D-Chain EGNN network denoises the coordinate sequence, conditioned on the topological structural embeddings from the Backward model.

### 4. Heuristic Volume Validation (Inference)
During live inference, simply generating a mathematical curve is not enough. The **Validity Evaluator** combined with `volume_utils` cross-references the synthesized 3D path against the raw biological TIFF volume. If the generated path does not align with high-intensity biological ridgelines (actual neurites), the connection is rejected.

## Parameters and Configuration

All hyperparameters are controlled globally via `config.yaml`.
**Key Parameters**:
- `dataloader.max_nodes`: Limits the maximum graph size loaded into GPU memory to prevent OOM errors.
- `forward_diffusion.beta_end`: Controls the maximum probability of an edge being dropped.
- `training.learning_rate`: Default set to `1e-4` with weight decay for regularization.
- `inference.max_joining_distance`: The maximum micron radius to search for candidate disconnected branches.
- `loss_weights`: Balances the BCE loss (topological correctness) vs the MSE loss (coordinate smoothness).

## Getting Started

### 1. Installation
Install the necessary PyTorch and scientific dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
The pipeline expects biological `.swc` files.
Ensure the paths in `config.yaml` point to your dataset:
```yaml
data:
  swc_dir: "/path/to/your/SWCs"
```

### 3. Training
To train the full co-dependent pipeline from scratch, simply execute the main orchestrator script:
```bash
python3 train.py
```
*Note: The trainer will automatically spawn DDP processes via `torch.multiprocessing` if multiple GPUs are detected.*
Training progress images (corrupted topologies vs. predicted connections) will be saved to `output/corrupted_disjoint_trees/` every epoch.

**Resuming Training:**
If you need to resume training from a specific checkpoint (e.g., epoch 100), you can use the dedicated resuming script which restores optimizer states and automatically appends to existing TensorBoard logs:
```bash
python3 resume_train.py
```

### 4. Inference
To run inference on a fragmented `.swc` and reconnect it using the trained models and a raw image volume:
```bash
python3 -m inference.main
```
The inference pipeline utilizes a highly optimized **CPU-Producer / Multi-GPU-Consumer** sliding window architecture to process arbitrarily large biological TIFF volumes. It connects sub-trees probabilistically and maps generated topologies precisely back to micron-scale SWC graphs, preserving original topological parent definitions.
