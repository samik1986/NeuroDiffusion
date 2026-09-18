# NeuroDiffusion Inference Pipeline

This folder contains the blazing fast inference pipeline for NeuroDiffusion, designed to reconstruct and extend neuronal subgraphs over massive 3D raw image volumes.

## Workflow

1. **Initialization**: The pipeline reads the TIF raw volume (e.g., 3.8GB) and the input SWC file containing disconnected subgraphs of neurites. It loads the `SequenceGenerator` (for sequential coordinate generation) and the `ValidityHeuristicEvaluator` (for scoring generated paths).
2. **Sliding Window Tiling**: Because the raw volume is too large for RAM, the image is processed in overlapping 3D chunks (default size: `256 x 256 x 64`).
3. **Graph Querying**: For each chunk, the pipeline queries the SWC graph for any endpoints that reside within the spatial boundaries of the chunk.
4. **Volume Pre-processing**: A fast Frangi vesselness filter (or Hessian-based ridgeness filter) extracts the intensity ridgelines and calculates the directional tangent field within the current chunk.
5. **Candidate Connection (Step 7)**: If multiple endpoints are near each other, the `SequenceGenerator` attempts to draw a path between them.
6. **Recursive Extension (Step 8)**: Individual endpoints can also be recursively extended coordinate-by-coordinate.
7. **Zero-Shot Validation**: Each generated path is scored by:
   - **Cost A**: The structural neural `ValidityHeuristicEvaluator`.
   - **Cost B**: The integral of the intensity ridgeness along the generated path, strongly penalized if the path's tangent direction misaligns with the local image intensity tangent.
8. **Accept/Reject**: The combined zero-shot cost determines if the path stays. If it bridges small intensity gaps but the overall cost is favorable, it is kept. Only one valid connection is kept between a pair of subgraphs.
9. **Visualization & Export**: The final reconstructed subgraphs are exported to `output/test/reconstructed.swc`. The results can be visualized in 3D using `napari`, differentiating original subgraphs from newly generated connections. Crucially, the export process faithfully maintains the exact parent-child topological links of the original SWC input, meaning the output contains actual continuous tree structures (skeletons) rather than disconnected point clouds.

## Assumptions

- The input TIF volume contains clear, albeit noisy, intensity ridgelines representing neurites.
- The voxel resolution is provided in `config.yaml` to ensure the generated coordinates scale properly.
- The models (Epoch 40 checkpoints) have been trained to generate scale-invariant coordinates.
- Subgraphs in the input SWC are topologically distinct connected components.
- Memory constraints necessitate overlapping window evaluation; connections crossing window borders are handled gracefully by overlap.

## Algorithm Details

- **Generative Method**: `SequenceGenerator` autoregressively predicts coordinates step-by-step.
- **Directional Alignment (Cost B)**: At each generated coordinate, the pipeline samples the Frangi vesselness orientation vectors. The dot product between the generation's step direction and the image's principle tangent orientation determines the directional penalty.
- **Speed Optimizations**: Multi-threading (`concurrent.futures.ThreadPoolExecutor`) is heavily utilized. Dask or joblib is used for parallel chunk evaluation, and numpy vectorization accelerates ray-tracing along paths.

## Implementation Details

- `volume_utils.py`: Contains the `SlidingWindowLoader` using memory mapping (`tifffile.memmap`), ensuring minimal RAM usage. Implements `skimage` Frangi filters.
- `graph_utils.py`: Uses an R-Tree or simple spatial hashing to rapidly query endpoints given a 3D bounding box.
- `evaluator.py`: Encapsulates the PyTorch model evaluation and numpy-based line integration.
- `generator.py`: Wraps the `SequenceGenerator.sample` into an iterative search algorithm.
- `main.py`: The entry orchestrator.

## Usage

```bash
python -m inference.main
```
