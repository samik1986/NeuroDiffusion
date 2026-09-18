# Models

This directory contains the core neural network architectures for the NeuroDiffusion pipeline. The generative pipeline is split into three main components: Topological (Backward) Diffusion, Coordinate (Sequence) Generation, and Validity Evaluation.

## Architectures

### 1. `backward_diffusion.py` (Backward Diffusion Model)
**Role:** Learns the topological rules of neuronal branching. Given a fragmented graph (with some edges dropped), it predicts which disconnected nodes should be connected.
**Architecture:** 
- A custom Graph Convolutional Network (GCN) layer that processes the surviving local neighborhood topologies.
- Uses `scatter_add` to dynamically aggregate messages from valid neighboring nodes without assuming a static adjacency matrix.
- Followed by a standard PyTorch `TransformerEncoder` (4 layers) that applies cross-node attention.
- Contains a global sum-pooling layer to extract a `context_emb` representation for the full graph, which is passed to the sequence generator.

### 2. `forward_diffusion.py` (Discrete Graph Diffusion)
**Role:** Acts as the noise-adder during training.
**Algorithm:**
- Implements a transition probability matrix. At $t=0$, the neuronal tree is pristine. As $t \to 100$, edges are incrementally dropped according to a linear or cosine schedule.
- The `forward_sample` method randomly drops edges to create the `noisy_edge_index` fragments that the Backward model must learn to fix.

### 3. `sequence_generator.py` (Sequence Generator Model)
**Role:** Generates continuous 3D spatial coordinates along the connected topologies.
**Architecture:**
- Uses a continuous Denoising Diffusion Probabilistic Model (DDPM) wrapped around a 1D-Chain Equivariant Graph Neural Network (EGNN).
- Employs Sinusoidal Positional Embeddings to denote the sequential order of branch nodes.
- At inference/generation, starts with $N(0, I)$ Gaussian noise and denoises it conditioned on the topological `context_emb` from the Backward Diffusion Model.

### 4. `heuristic_evaluator.py` (Validity Evaluator)
**Role:** Evaluates whether a generated continuous path physically intersects with existing geometry in the volume.
**Architecture:**
- A 1D-Chain EGNN that processes generated paths and outputs an overlap penalty score to discourage mathematically valid but physically impossible (colliding) structures.

## Assumptions
- **Target Leakage Avoidance:** The models here do *not* receive ground-truth Laplacian Positional Encodings (`lap_pe`), forcing them to learn true topological rules through Message Passing rather than reading "cheat codes" from eigenvectors.
- **Normalization:** The `SequenceGenerator` assumes input coordinates are normalized to $[-1, 1]$ relative to the max bounding box dimension of the neuronal branch to ensure the standard DDPM Gaussian noise scale is mathematically sound.
