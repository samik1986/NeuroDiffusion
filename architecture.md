# NeuroDiffusion Architecture

This document outlines the architecture for the two primary generation models in the pipeline: the **Backward Diffusion Model** (for topological edge prediction) and the **Sequence Generator** (for generating the exact morphological sequences/coordinates).

## 1. Backward Diffusion Model (Topology Reconstruction)

The backward diffusion model is a specialized Graph Transformer. It takes disjoint, fragmented neuronal sub-trees (produced by the forward diffusion corruption) and uses global self-attention to exchange context between these fragments. It then predicts the probability of candidate edges to bridge the missing gaps.

```mermaid
graph TD
    %% Inputs
    X[Node Coordinates N, 3] --> NodeMLP[Node MLP]
    Edges[Noisy/Fragmented Edges 2, E] --> MP[Message Passing over Sub-trees]
    T[Timestep t] --> TimeMLP[Time MLP Sinusoidal]
    Candidates[Candidate Edges 2, E_cand] --> EdgeHead

    %% Processing
    NodeMLP --> MP
    MP --> AddTime
    TimeMLP --> AddTime[Add Time Embedding]
    
    AddTime --> Pad[Pad to Batches for Transformer]
    Pad --> Trans[Transformer Encoder Layer]
    Trans --> Flatten[Un-pad / Flatten]
    
    %% Prediction Head
    Flatten --> ExtractSrcDst[Extract Source & Dest Nodes for Candidates]
    ExtractSrcDst --> ConcatPairs[Concatenate Src, Dst]
    ConcatPairs --> EdgeHead[Edge Prediction Head MLP]
    
    EdgeHead --> OutputLogits[Edge Logits]
    Flatten --> OutputContext[Graph Context Embeddings]
    
    style OutputLogits fill:#f9f,stroke:#333,stroke-width:2px
    style OutputContext fill:#f9f,stroke:#333,stroke-width:2px
```

## 2. Sequence Generator (Morphology Generation)

The sequence generator acts as a continuous DDPM (Denoising Diffusion Probabilistic Model). Rather than predicting edges, it generates the exact 3D coordinates along a neuronal branch. It is conditioned on the graph context extracted from the Backward Diffusion Model.

```mermaid
graph TD
    %% Inputs
    NoisyCoords[Noisy Sequence Coords x_t B, SeqLen, 3] --> CoordProj[Coordinate Projection Linear]
    T[Timestep t] --> TimeMLP[Time MLP Sinusoidal]
    Context1[Context 1] --> ConcatCtx[Concatenate Contexts]
    Context2[Context 2 / Empty] --> ConcatCtx
    
    %% Processing
    ConcatCtx --> CtxProj[Context Projection Linear]
    
    CoordProj --> AddAll
    TimeMLP --> AddAll
    CtxProj --> AddAll
    PosEmb[1D Positional Embedding] --> AddAll[Sum Representations: h = x + t + ctx + pos]
    
    AddAll --> Trans[Transformer Encoder Layer]
    
    %% Prediction Heads
    Trans --> NoiseHead[Noise Prediction Head]
    Trans --> TypeHead[Node Type Head]
    
    NoiseHead --> OutputNoise[Predicted Noise epsilon_theta]
    TypeHead --> OutputType[Predicted Node Types]
    
    style OutputNoise fill:#f9f,stroke:#333,stroke-width:2px
    style OutputType fill:#f9f,stroke:#333,stroke-width:2px
```

## How they interact
1. The **Backward Diffusion Model** evaluates a fragmented graph, predicts how it connects, and produces a highly descriptive `Graph Context Embedding` for those connected components.
2. The **Sequence Generator** takes that `Graph Context Embedding` as its conditional input (`Context 1` and `Context 2`), ensuring that the 3D morphology it generates natively respects the overall topological structure of the neuronal tree.
