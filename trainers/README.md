# Trainers

This module contains the encapsulated PyTorch optimization loops for each of the core neural network architectures. Separating the training logic allows for clean, modular execution inside the global `train.py` orchestrator.

## Components

### 1. `diffusion_trainer.py`
**Role:** Manages the optimization of the Backward Diffusion (Graph) Model.
**Key Mechanics:**
- **Automatic Mixed Precision (AMP):** Utilizes `torch.autocast` and `GradScaler` to significantly speed up training on modern GPUs (e.g., H100s) while reducing VRAM usage.
- **Intra-batch Negative Sampling:** Automatically generates topologically invalid (negative) candidate edges directly within the GPU loop to create a balanced Binary Cross-Entropy (BCE) classification task for the model to predict which edges are true vs. false.
- **DDP Compatibility:** Written to seamlessly support Distributed Data Parallel processing.

### 2. `generator_trainer.py`
**Role:** Manages the optimization of the Coordinate Sequence Generator Model.
**Key Mechanics:**
- Computes both continuous Mean Squared Error (MSE) loss for the 3D coordinates and Cross-Entropy loss for the node branch types (e.g., continue, branch, terminate).

### 3. `evaluator_trainer.py` (Heuristic Trainer)
**Role:** Manages the optimization of the physical validity heuristic.
**Key Mechanics:**
- Computes collision/intersection penalties between the generated 3D neuronal branches and the background volume constraints.

## Integration
All trainers are instantiated inside `train.py`, which iterates over the `DataLoader` and invokes their respective `train_step()` and `val_step()` methods sequentially during the co-training phase.
