# NeuroDiffusion Algorithm Formulation & Tuning Guide

This document provides a mathematical and heuristic formulation of each step in the NeuroDiffusion pipeline, along with a guide on tuning hyperparameters. It covers both the training regime and the recent sliding-window volumetric inference architecture.

## 1. Forward Diffusion (Graph Fragmentation)
**Formulation:** We model the destruction of neuronal trees as a discrete diffusion process on the edge set. At time $t=0$, the graph $G$ has fully connected trees. At time $t$, each edge is dropped according to a Bernoulli distribution parameterized by a noise schedule $\bar{\alpha}_t$. This breaks complete neurons into the disjoint sub-trees seen in raw fragmented SWC data.
**Parameters to Tune:**
- `forward_diffusion.beta_end`: Controls the maximum noise rate. A higher `beta_end` results in more severely fragmented trees during training. If the model struggles to connect distant fragments, increase this value to expose it to harder, more fractured examples during training.

## 2. Backward Diffusion (Topological Prediction)
**Formulation:** The model acts as a hybrid EGNN and Graph Transformer. It takes disjoint sub-trees and predicts a probability $p_{uv} = \sigma(\text{MLP}([h_u, h_v]))$ that an edge should exist between nodes $u$ and $v$ from different fragments, extracting a dense topological context embedding for the sub-trees.
**Parameters to Tune:**
- `training.learning_rate`: Default `1e-4`. If topological predictions are unstable or oscillating, reduce to `5e-5`.

## 3. Sequence Generation (Morphology Synthesis)
**Formulation:** Given contexts $c_1$ and $c_2$ from two disjoint sub-trees, a continuous DDPM synthesizes a 3D coordinate sequence $X_{1:L}$. The model optimizes the objective $L_{\text{coord}} = ||\epsilon - \epsilon_\theta(x_t, t, c_1, c_2)||^2$, denoising the sequence from pure Gaussian noise into a smooth biological curve.
**Parameters to Tune:**
- `loss_weights.coord`: (Default 10.0) Increasing this weight forces the model to prioritize smooth, accurate morphological paths over purely topological correctness.

## 4. Heuristic & Intensity Volume Evaluator (Inference Cost)
**Formulation:** During inference, we evaluate a generated path $X$ connecting trees $T_1$ and $T_2$. The cost $C$ is a composite of the Neural Gap (predicted by the Heuristic Evaluator) and the Intensity Path Integral (computed from the raw biological TIFF volume using Vesselness and Tangent metrics).
$$ C = \alpha \cdot \text{Gap}(T_1, T_2, X) - \beta \cdot \frac{1}{|X|} \sum_{i} \text{Vesselness}(x_i) \cdot |\langle \text{Tangent}(x_i), \text{StepDir}(x_i) \rangle| $$
**Parameters to Tune:**
- `alpha` / `beta` in `inference/evaluator.py`: 
  - Increase `alpha` if the model connects physically proximal but structurally illogical branches.
  - Increase `beta` if the model ignores the actual biological image data and hallucinates connections through empty space.
- `max_joining_distance`: The search radius for candidate branches. Increasing this finds more candidates but drastically increases inference time and potential false positive topological errors.

## 5. Sliding Window Volumetric Inference Architecture
**Formulation:** To process massive biological TIFFs (which cannot fit in VRAM), a CPU producer process loads overlapping sliding windows (e.g., `(64, 256, 256)`). It computes Sato/Frangi Vesselness and Tangent volumes on the CPU. A queue distributes these windows to multiple GPU consumers which concurrently evaluate candidate endpoints inside the window. The generated connections are mathematically transformed back from the voxel coordinate space to the raw physical scale (microns), preserving the exact topology and parent structure of the original SWC nodes, resulting in a single unified graph of fully reconstructed neurons.
