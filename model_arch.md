# Layer-wise Model Architecture

This document explicitly unrolls all major neural network blocks (EGNNs, Transformers, MLPs) down to their primitive PyTorch layers (Linear, SiLU, BatchNorm, etc.), providing a complete node-to-node operational trace for the NeuroDiffusion models.

## 1. EGNN Layer Unrolled (Equivariant Graph Neural Network)
This is the core local message-passing layer utilized across Backward Diffusion, Sequence Generation, and Heuristic Evaluation.

```mermaid
graph TD
    %% Inputs
    H_in[Node Features h N,128] --> ConcatEdge
    X_in[Node Coords x N,3] --> DistCalc
    EdgeIdx[Edge Index 2,E] --> ConcatEdge
    
    %% Compute Distance
    DistCalc[Radial Dist: sum x_i - x_j^2] --> ConcatEdge
    
    %% Edge Message
    ConcatEdge[Concat h_i, h_j, dist E,257] --> ELinear1[Linear 257, 256]
    ELinear1 --> ESiLU1[SiLU]
    ESiLU1 --> ELinear2[Linear 256, 128]
    ELinear2 --> ESiLU2[SiLU]
    ESiLU2 --> M_ij[Edge Message m_ij E,128]
    
    %% Coord Update (Equivariant)
    M_ij --> CLinear1[Linear 128, 128]
    CLinear1 --> CSiLU1[SiLU]
    CSiLU1 --> CLinear2[Linear 128, 1]
    CLinear2 --> CTanh[Tanh * 10.0]
    CTanh --> CoordWeight[Coord Weight E,1]
    
    DistCalc --> Normalize[coord_diff / dist]
    Normalize --> ScaleDiff[Scale: diff * coord_weight]
    ScaleDiff --> ScatterAddX[Scatter Add degree normalized]
    ScatterAddX --> X_out[Updated Coords x_out = x + trans N,3]
    
    %% Node Update (Invariant)
    M_ij --> ScatterAddH[Scatter Add m_ij over neighbors]
    ScatterAddH --> ConcatNode[Concat h, m_i N,256]
    
    ConcatNode --> NLinear1[Linear 256, 256]
    NLinear1 --> NSiLU1[SiLU]
    NSiLU1 --> NLinear2[Linear 256, 128]
    NLinear2 --> H_out[Updated Features h_out = h + update N,128]
```

## 2. Backward Diffusion Model

```mermaid
graph TD
    %% Time Embed
    T[Timestep] --> T_Emb[Sinusoidal 128]
    T_Emb --> TLinear1[Linear 128,128]
    TLinear1 --> TSiLU[SiLU]
    TSiLU --> TLinear2[Linear 128,128]
    TLinear2 --> T_Out[Time Embeddings N,128]
    
    X[Node Coordinates N,3] --> EGNN_Block
    Edges[Noisy Edges 2,E] --> EGNN_Block
    T_Out --> EGNN_Block
    
    %% EGNN Unrolled stack
    EGNN_Block[4x Unrolled EGNN Layers] --> H_Inv[Invariant Features N,128]
    
    %% Transformer Encoder Unrolled
    H_Inv --> PadBatch[Pad Sequence B,MaxN,128]
    PadBatch --> QKV[QKV Proj: Linear 128, 3x128]
    QKV --> MHA[Multi-Head Attn: 4 heads]
    MHA --> AddNorm1[Add & LayerNorm]
    AddNorm1 --> FFN1[Linear 128, 512]
    FFN1 --> FFN_ReLU[ReLU]
    FFN_ReLU --> FFN2[Linear 512, 128]
    FFN2 --> AddNorm2[Add & LayerNorm]
    
    AddNorm2 --> Repeat[Repeat 4x]
    Repeat --> Unpad[Flatten BxMaxN -> N,128]
    
    %% Edge Head
    Unpad --> ExtractPairs[Extract Src & Dst for Candidates]
    ExtractPairs --> ConcatHead[Concat Src, Dst, Scale-Inv Log Dist 257]
    ConcatHead --> HeadLin1[Linear 257, 128]
    HeadLin1 --> HeadSiLU[SiLU]
    HeadSiLU --> HeadLin2[Linear 128, 1]
    HeadLin2 --> Logits[Edge Logits]
```

## 3. Sequence Generator

```mermaid
graph TD
    %% Time Embed
    T[Timestep] --> T_Emb[Sinusoidal 128]
    T_Emb --> TLinear1[Linear 128,256]
    TLinear1 --> TGELU[GELU]
    TGELU --> TLinear2[Linear 256,128]
    
    %% Context Embed
    Ctx1[Context 1] --> ConcatCtx[Concat 256]
    Ctx2[Context 2] --> ConcatCtx
    ConcatCtx --> CtxLin[Linear 256, 128]
    
    %% Pos Embed
    Pos[0..L] --> PosEmb[Sinusoidal 128]
    
    %% Init features
    TLinear2 --> SumH
    CtxLin --> SumH
    PosEmb --> SumH[h_0 = t + ctx + pos N,128]
    
    X[Noisy Coords x_t N,3] --> EGNN_Chain
    SumH --> EGNN_Chain
    
    %% 1D Chain
    EGNN_Chain[4x Unrolled EGNN Layers on 1D Path Edges] --> H_Out[Invariant Features N,128]
    EGNN_Chain --> X_Out[Equivariant Coords N,3]
    
    %% Prediction Heads
    X_Out --> SubNoise[Noise: epsilon = x_out - x_t]
    
    H_Out --> TypeLin[Linear 128, 3]
    TypeLin --> Types[Node Types Logits]
```

## 4. Heuristic Evaluator

```mermaid
graph TD
    %% Time Embed
    T[Timestep] --> T_Emb[Sinusoidal 128]
    T_Emb --> TLinear1[Linear 128,128]
    TLinear1 --> TSiLU[SiLU]
    
    %% Type Embed
    Types[Path Types N,3] --> TypeLin[Linear 3, 128]
    
    %% EGNN Path Encoder
    X[Path Coords N,3] --> EGNN_Path
    TypeLin --> EGNN_Path
    
    EGNN_Path[2x Unrolled EGNN Layers on 1D Path Edges] --> H_Out[Invariant Path Features N,128]
    
    %% Pooling
    H_Out --> MeanPool[Masked Mean Pooling B,128]
    
    %% Combine
    MeanPool --> ConcatAll[Concat T1, T2, Path, Time B,512]
    
    %% Classifier
    ConcatAll --> CLin1[Linear 512, 256]
    CLin1 --> CBatchNorm[BatchNorm1d 256]
    CBatchNorm --> CSiLU1[SiLU]
    CSiLU1 --> Drop[Dropout 0.2]
    Drop --> CLin2[Linear 256, 128]
    CLin2 --> CSiLU2[SiLU]
    CSiLU2 --> CLin3[Linear 128, 1]
    CLin3 --> Gap[Predicted Structural Gap]
```
