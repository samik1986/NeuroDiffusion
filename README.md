# NeuroDiffusion Algorithm Design Plan

## Overview
This plan outlines the design of a diffusion-based model for reconstructing 3D neurons from fragmented SWC data, based on the specifications in `Steps.txt`. The algorithm focuses on generating scale, rotation, and translation-invariant representations, simulating fragmentation through forward diffusion, and learning to reconstruct the full tree via backward diffusion.

## Resolved Decisions
> [!NOTE]
> 1. The model will explicitly predict the continuous 3D coordinates of new fragments (generative trajectory).
> 2. For invariance, we will use relative coordinates (angles and distances between parent-child nodes) as inputs.
> 3. Standard Denoising Diffusion Probabilistic Models (DDPM) combined with discrete edge manipulation is the chosen approach.

## Proposed Algorithm & Mathematical Formulation

### 1. Data Representation & Preprocessing
*   **Input**: SWC files from `../NeuroVLM/SWCs`. Somas (type 1) are filtered out.
*   **Representation**: A neuron is represented as a directed graph $G = (V, E)$ where $V \in \mathbb{R}^{N \times 3}$ are node coordinates and $E$ are edges representing neuronal branches. Each node has a type identifier (endpoint, branchpoint, continuation).
*   **Invariance Formulation**:
    *   *Translation*: Center the entire neuron or each fragment to its centroid ($x'_i = x_i - \bar{x}$).
    *   *Scale*: Normalize the coordinates such that the maximum bounding box dimension or variance is 1.
    *   *Rotation*: The model will use invariant geometric features (relative distances between connected nodes, and angles between segments) as inputs to naturally achieve rotational invariance.

### 2. Forward Diffusion (Fragmentation)
Instead of adding Gaussian noise to coordinates, we apply a discrete diffusion process by progressively masking/removing edges to break the tree into sub-trees.
*   **Mathematical Formulation**: Let $E_0$ be the true edges. At timestep $t \in [1, T]$, we drop edges with a schedule $\beta_t$.
*   $q(E_t | E_{t-1}) = \text{Categorical}(E_t; (1-\beta_t)E_{t-1} + \beta_t \cdot \text{noise})$
*   As $t \rightarrow T$, the neuron breaks down into smaller, disconnected trees (fragments) in random locations.
*   **Parameters**:
    *   $T$: Total diffusion steps. Higher $T$ provides smoother transitions but longer training.
    *   $\beta_t$: Variance schedule (controls how aggressively trees are broken at each step). A linear or cosine schedule can be used.

### 3. Backward Diffusion (Reconstruction)
The model learns to reverse the fragmentation by predicting missing connections between fragments to form a single neuron.
*   **Heuristic Formulation**: The model predicts a connection probability $P(e_{ij} = 1 | G_t)$ between two nodes $i$ and $j$. Valid connection heuristics are enforced (endpoint-to-endpoint, branchpoint-to-endpoint).
*   **Network Architecture**: A Graph Neural Network (GNN) or Graph Transformer that processes $G_t$ and outputs edge logits.
*   **Mathematical Formulation**: $p_\theta(E_{t-1} | E_t) = \mathcal{N}_\theta(E_t, t)$ where $\mathcal{N}_\theta$ predicts the edges added back to reduce the number of fragments.
*   **Parameters**:
    *   Network Layers/Heads: Determines the receptive field and capacity to learn complex structural heuristics.

### 4. Sequence Generation (Extension)
To predict the next sequences or extend fragments from endpoints or branchpoints:
*   Given an endpoint or branchpoint as a conditioning seed, the model autoregressively samples the next node's relative coordinate $\Delta x$ from the learned distribution $p_\theta(\Delta x_t | \text{context})$.
*   *Branching within Sequence*: During the generation of a sequence path between two points, the model may also generate new branchpoints internally if such structures are supported and sampled from the learned data distribution.
*   **Parameters**:
    *   Temperature ($\tau$): Controls the randomness of the generated sequences during sampling. Higher $\tau$ yields more diverse branch generation.

### 5. Cost Function (Loss)
The objective is to minimize the number of fragments, which mathematically equates to maximizing the likelihood of predicting the correct joining edges and next-sequence coordinates.
*   **Formulation**: $\mathcal{L} = \mathcal{L}_{edge} + \lambda \mathcal{L}_{coord}$
    *   $\mathcal{L}_{edge}$: Binary Cross-Entropy (BCE) between predicted edge probabilities and true edges from $E_0$. Minimizing this naturally reduces the number of disconnected fragments by penalizing missed connections.
    *   $\mathcal{L}_{coord}$: Mean Squared Error (MSE) or Gaussian Negative Log-Likelihood for predicting the 3D coordinates of generated next sequences.
*   **Parameters**:
    *   $\lambda$: Weighting factor balancing the connection prediction (joining fragments) versus continuous coordinate generation (extending fragments). Tuning this dictates whether the model prioritizes structural connectivity or precise spatial trajectories.

### 6. Validity Heuristic for Generated Paths
To evaluate whether a generated sequence connecting two disjoint trees is valid, a learned heuristic (discriminator/evaluator function) $H(T_1, T_2, P)$ will be introduced, where $T_1$ and $T_2$ are the disjoint trees and $P$ is the generated path.
*   **Heuristic Formulation**: The function $H$ outputs a validity score $s \in [0, 1]$ indicating the biological plausibility of the connection.
*   **Inputs**: The combined graph of $T_1$, $T_2$, and the sequence path $P$ parameterized by its 3D coordinates, branching angles, and lengths.
*   **Evaluation Criteria**: 
    1.  **Distribution Alignment**: Is the path $P$ sampled correctly from the expected distribution of SWC connections?
    2.  **Geometric Smoothness**: Do the angles at the connection points (endpoints/branchpoints) follow biological continuity without sharp, unnatural turns?
    3.  **Endpoint/Branchpoint Matching**: Does the sequence properly terminate at a valid endpoint or branchpoint of the target tree?
*   **Network Output**: This heuristic can be trained as a separate binary classifier or as an auxiliary head on the DDPM, trained on true biological paths (label 1) and random/incorrect paths (label 0).

### 7. Inference Pipeline (Joining Fragmented Trees)
The inference stage utilizes the trained generative models (Backward Diffusion, Sequence Generator) and the Heuristic Evaluator alongside raw 3D volume intensities to reconnect fragmented neurons.

#### Pipeline Architecture & Flow
1. **Graph Separation**: Given an input SWC with multiple disconnected fragments, we partition the SWC graph into isolated sub-graphs (trees) $\mathcal{T} = \{T_1, T_2, \dots, T_N\}$.
2. **Nearest Neighbor Search**: For the endpoints of a target tree $T_i$, we search for spatially close candidate nodes in other trees $T_j \in \mathcal{T} \setminus \{T_i\}$ within a maximum distance $d_{max}$.
3. **Generative Connection**:
   - The **Backward Diffusion GNN** processes $T_i$ and $T_j$ to provide a structural connectivity logit $P(e_{ij} = 1 | T_i, T_j)$.
   - The **Sequence Generator** autoregressively draws a path $P = \{p_1, p_2, \dots, p_k\}$ in 3D space connecting the endpoint of $T_i$ to $T_j$, potentially generating internal branchpoints according to the learned distribution.
4. **Heuristic Evaluation**:
   - The neural **Validity Heuristic Evaluator** calculates $H(T_i, T_j, P) \in [0, 1]$ representing the biological likelihood of the generated sequence structure.
5. **Volume Ridgeline Evaluation**:
   - The raw 3D multichannel TIFF volume $V(x,y,z)$ is loaded, scaled by voxel resolutions (e.g. $0.1102 \times 0.1102 \times 0.5 \mu m$).
   - We extract the intensity array $I(P) = \{V(p_1), V(p_2), \dots, V(p_k)\}$ by interpolating the raw volume at the exact 3D generated coordinates.
   - We enforce an intensity ridgeline condition: $\frac{1}{k} \sum_{m=1}^{k} V(p_m) > \tau$, where $\tau$ is an intensity threshold parameter (e.g., $80^{th}$ percentile of local intensity).
6. **Acceptance Criterion**:
   - The path is physically joined to the SWC graph if it satisfies both the neural heuristic and the biological volume intensity constraints:
   $$A(P) = \mathbb{1}[H(T_i, T_j, P) > 0.5] \land \mathbb{1}[\text{mean}(I(P)) > \tau]$$
7. **Output**: The connected sub-graphs merge, producing a cohesive SWC representing partial/full neuron reconstructions.

## Verification Plan
1.  **Unit Testing**: Verify data loaders correctly parse SWC files, ignore somas, and apply translation/scale/rotation invariance correctly.
2.  **Forward Process Visualization**: Visualize the graph at $t=0, t=T/2, t=T$ to ensure the neuron is breaking into logical fragments (trees) and not just isolated points.
3.  **Model Training**: Train on a small subset of `../NeuroVLM/SWCs` and monitor the loss $\mathcal{L}$ to ensure it converges and the number of output fragments decreases over epochs.
4.  **Inference Validation**: Ensure the final orchestrator successfully extracts trees, runs generative paths, verifies intensities over `F0046_multichannel_cmle_ch03.tif`, and merges accepted SWC outputs.
