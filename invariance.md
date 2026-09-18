# NeuroDiffusion: Invariance Properties & Theoretical FAQ

This document captures key theoretical questions regarding the invariance properties and reliability of the NeuroDiffusion pipeline.

## 1. Do the models use Laplacian Positional Encodings (PE) or Sinusoidal Embeddings?

We exclusively use **Sinusoidal** position embeddings throughout the models.

**Laplacian Positional Encodings (`lap_pe`) were intentionally removed** from the pipeline for two critical reasons:
1. **Target Leakage:** Providing the Laplacian eigenvectors allowed the Backward Diffusion Model to essentially "cheat" and memorize the global topology rather than learning the actual 3D geometric rules through local message passing. 
2. **Speed Bottleneck:** Computing the sparse eigendecomposition for the Laplacian was causing massive CPU bottlenecks in the DataLoaders.

Instead:
- The **Sequence Generator** uses standard **1D Sinusoidal Positional Embeddings** to denote the sequential order of branch nodes.
- The **Timestep Embeddings** across all models (Backward, Generator, Heuristic) use **Sinusoidal Embeddings**. 

The EGNNs rely purely on the explicit 3D geometry (`coord_diff`) and continuous message passing rather than needing pre-computed Laplacian graphs!

---

## 2. What is the degree of invariance of the model?

The NeuroDiffusion pipeline achieves a very high degree of invariance, specifically designed to handle the geometric realities of 3D biology:

### Translation Invariance
- **Absolute Position Ignored:** The global absolute position of a neuron in 3D space is ignored. In the DataLoader, coordinates are mean-centered (`coords - centroid`) before processing.
- **EGNN Mechanism:** Throughout the models, the EGNN layers compute messages strictly using relative coordinate differences ($x_i - x_j$) and squared distances ($||x_i - x_j||^2$), making all internal hidden features mathematically blind to global translation.

### Rotation & Reflection Invariance (E(3) Group)
- **Invariant Node Features (O(3) Invariant):** The node embeddings ($h$) produced by the EGNN layers are strictly invariant to 3D rotations and reflections. Because the EGNN updates $h$ using only scalar radial distances, rotating the entire brain volume yields the exact same $h$ embeddings.
- **Equivariant Coordinate Generation (O(3) Equivariant):** The Sequence Generator predicts coordinate noise ($\Delta x$). Because it uses an EGNN, this prediction is **equivariant**. Rotating the input sub-trees by 45 degrees guarantees the model's generated sequence perfectly rotates by that exact same 45 degrees.

### Scale Invariance
- **Topological Scale Independence:** In the Backward Diffusion Model, when evaluating the distance between two fragments to predict an edge, the model explicitly computes the spatial variance (`scale_sq`) of the given graph. It normalizes the gap distance (`sq_dist / scale_sq`) before feeding it to the Edge Head. The model confidently connects a 5-micron gap in a tiny graph just as it would a 500-micron gap in a massive graph.
- **Morphological Scale Independence:** The DataLoader scales all branch coordinates into a normalized $[-1, 1]$ bounding box, ensuring the DDPM's standard Gaussian noise schedule $N(0, I)$ works flawlessly regardless of the raw biological micron scale.

### Permutation Invariance
- **Graph Order Independence:** The Backward Diffusion model relies on permutation-invariant `scatter_add` operations to aggregate neighborhood messages and a symmetric `TransformerEncoder` to share global context. Shuffling the order of the nodes in the SWC file array will not change the predicted topology.

---

## 3. How does permutation invariance affect the results?

Permutation invariance has a massive, highly practical impact on the reliability and robustness of the results:

1. **Immunity to SWC Formatting Differences:** Depending on the tracing software used (Vaa3D, NeuTube, ImageJ), SWC nodes might be sorted topologically, spatially, or randomly. Because the model is permutation invariant, it evaluates relationships (edges), producing the exact same topological predictions regardless of how the file is sorted.
2. **Prevention of Index Memorization:** Without permutation invariance, a model might memorize artifactual index patterns (e.g., *"Node Index #5 usually connects to Node Index #6"*). By using permutation-invariant operations, the model is strictly forced to learn the actual **3D geometry and structural rules**.
3. **Generalization to Unseen Graph Sizes:** Because the model aggregates features without caring about array positions, it can dynamically adapt to massive graphs during inference. A model trained on 200-node fragments can seamlessly generalize to a biological volume containing 50,000 fragmented nodes.

---

## 4. Can this process connect two totally disjoint graphs which should not be connected?

While there is always a statistical risk of "false positives" in generative models, NeuroDiffusion uses a strict **three-layered gauntlet** during inference to prevent accidentally connecting unrelated neurons or noise fragments:

1. **The Topological Gate (Backward Model):** Evaluates the local geometry of the disjoint endpoints. If the fragments are structurally incompatible (pointing in different directions or lacking complementary context), the model outputs a near-zero probability.
2. **The Structural Gate (Heuristic Evaluator):** If the Sequence Generator attempts to draw a path, the Heuristic Evaluator discriminates if the generated 3D curve is biologically possible. Sharp, unnatural hairpin turns receive a massive "Neural Gap" penalty (Cost A).
3. **The Physical Ground-Truth Gate (Intensity Cost):** This is the ultimate safeguard. The generated path is verified against the **raw biological image volume (the TIFF stack)** via a Path Integral (Cost B). Connections cutting through dark space (low Vesselness) or travelling perpendicular to physical neurites (Tangent Misalignment) are heavily penalized and rejected.

For a false connection to occur, the topologies must accidentally align perfectly, the generated path must look morphologically natural, **and** there must be a coincidentally bright physical artifact perfectly bridging the two fragments in the raw microscope image.
