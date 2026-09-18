---
marp: true
theme: default
paginate: true
---

# NeuroDiffusion: Pipeline Overview
**Generative Reconstruction of 3D Neuronal Topologies**

---

## The Biological Problem

- **Raw Data Reality:** Biological `.swc` datasets trace 3D neurons, but they are frequently highly **fragmented** due to imaging artifacts, staining inconsistencies, and tracing errors.
- **The Consequence:** A single biological neuron is represented as dozens of disconnected sub-trees (fragments) floating in 3D space.
- **The Goal:** To mathematically infer the missing topological connections (edges) and synthesize the smooth 3D morphological sequence (coordinates) that bridges them.

---

## The NeuroDiffusion Solution

We frame this reconstruction as reversing a diffusion (fragmentation) process:
1. **Forward Diffusion:** Systematically destroy edges of pristine trees to simulate tracing errors.
2. **Backward Diffusion:** Predict *which* fragments should connect (Topology).
3. **Sequence Generation:** Synthesize *how* they connect in 3D space (Morphology).
4. **Heuristic Evaluation:** Zero-shot validation against raw image intensity constraints.

---

## Universal Invariances

The entire pipeline is built upon deep geometric invariances required for 3D biological data:
- **E(3) Invariance:** Equivariant Graph Neural Networks (EGNNs) process nodes entirely via relative radial distances ($||x_i - x_j||^2$). The model is structurally blind to global translations and is perfectly invariant to 3D rotations/reflections.
- **Scale Invariance:** Connections are scored using a normalized gap distance ($\text{distance}^2 / \text{graph\_variance}$). Standard DDPM generation is done in a normalized bounding box $[-1, 1]$.
- **Permutation Invariance:** The models utilize `scatter_add` for local processing and symmetric global attention, ensuring immunity to how the SWC text file is formatted or ordered.

---

## Technical Architecture Stack

The codebase operates via orchestrated, modular PyTorch loops:
- **Models:** Built on customized EGNNs, continuous DDPMs, and Transformer Encoders.
- **Trainers:** Isolated AMP (Automatic Mixed Precision) and DDP (Distributed Data Parallel) optimization loops.
- **Zero-Shot Evaluators:** CPU-Producer / Multi-GPU-Consumer sliding windows over massive TIFF volumes utilizing Frangi/Sato vesselness filtering.
