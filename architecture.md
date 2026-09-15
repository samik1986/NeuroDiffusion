# NeuroDiffusion Architecture

This document outlines the architecture for the four primary models in the NeuroDiffusion pipeline:
1. **Forward Diffusion Model** (Edge corruption process)
2. **Backward Diffusion Model** (Topological edge prediction)
3. **Sequence Generator** (Morphological sequence generation)
4. **Heuristic Evaluator** (Validity scoring)

---

## 1. Forward Diffusion Model (Discrete Edge Corruption)

The discrete forward diffusion process simulates the fragmentation of neuronal trees. It takes a fully connected graph (tree) and progressively drops edges over time $t$ according to a cosine or linear noise schedule. 

```mermaid
graph TD
    %% Inputs
    Edges[True Graph Edges 2, E] --> SampleDrop[Bernoulli Sampling: Drop Edges]
    T[Timestep t] --> Schedule[Compute Alpha_bar_t]
    
    %% Processing
    Schedule --> SampleDrop
    SampleDrop --> OutputNoisy[Noisy/Fragmented Edges 2, E_noisy]
    
    style OutputNoisy fill:#f9f,stroke:#333,stroke-width:2px
```

---

## 2. Backward Diffusion Model (Topology Reconstruction)

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

---

## 3. Sequence Generator (Morphology Generation)

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

---

## 4. Heuristic Evaluator (Validity Scoring)

The Validity Heuristic Evaluator checks whether a generated morphological sequence correctly and biologically connects two sub-trees. It combines the contextual embeddings of the sub-trees, the sequence itself, and the diffusion timestep to produce a validity score.

```mermaid
graph TD
    %% Inputs
    T1[Context T1 B, CtxDim] --> ConcatAll
    T2[Context T2 B, CtxDim] --> ConcatAll
    PathC[Path Coords B, SeqLen, 3] --> ConcatPath
    PathT[Path Types B, SeqLen, 3] --> ConcatPath
    T[Timestep t] --> TimeMLP[Time MLP Sinusoidal]
    
    %% Processing
    ConcatPath[Concat Coords + Types] --> BiLSTM[BiLSTM Path Encoder]
    BiLSTM --> PathEmb[BiLSTM Path Embedding]
    
    PathEmb --> ConcatAll[Concatenate All Features]
    TimeMLP --> ConcatAll
    
    ConcatAll --> Classifier[Classifier MLP with Dropout]
    Classifier --> Logit[Validity Logit / Probability]
    
    style Logit fill:#f9f,stroke:#333,stroke-width:2px
```

---

## Overall Pipeline Flow
1. **Forward Diffusion Model** tears apart a complete neuronal graph.
2. The **Backward Diffusion Model** evaluates this fragmented graph, predicts how it connects (Topology), and produces a highly descriptive `Graph Context Embedding` for those connected components.
3. The **Sequence Generator** takes that `Graph Context Embedding` as its conditional input (`Context 1` and `Context 2`) and iteratively denoises a path to generate the 3D morphology between those sub-trees.
4. The **Heuristic Evaluator** acts as a final filter, taking the generated path and the original context to reject biologically impossible or topologically invalid connections.
