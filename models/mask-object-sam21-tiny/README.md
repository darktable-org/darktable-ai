# SAM 2.1 (Hiera Tiny)

Segment Anything Model 2.1 — tiny variant with Hiera encoder.
Multi-mask output mode (3 masks per prompt).

## Source

- Repository: https://github.com/facebookresearch/sam2
- Paper: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) (2024)
- License: Apache-2.0

## Architecture

Hiera Tiny encoder with SAM 2 mask decoder and high-resolution feature
projections (conv_s0, conv_s1). Bakes high-res feature convolutions into the
encoder so that decoder receives pre-projected features.

## ONNX Models

Split into encoder (image features) and decoder (prompt fusion + mask prediction).

### encoder.onnx

| Property    | Value                                           |
|-------------|--------------------------------------------------|
| Input       | `image` — float32 [1, 3, 1024, 1024]            |
| Output 1    | `high_res_feats_0` — float32 [1, 32, 256, 256]  |
| Output 2    | `high_res_feats_1` — float32 [1, 64, 128, 128]  |
| Output 3    | `image_embed` — float32 [1, 256, 64, 64]        |
| Resolution  | Fixed 1024x1024                                  |
| Normalize   | ImageNet mean/std                                |

### decoder.onnx

| Property    | Value                                           |
|-------------|--------------------------------------------------|
| Input 1     | `image_embed` — float32 [1, 256, 64, 64]        |
| Input 2     | `high_res_feats_0` — float32 [1, 32, 256, 256]  |
| Input 3     | `high_res_feats_1` — float32 [1, 64, 128, 128]  |
| Input 4     | `point_coords` — float32 [1, N, 2]              |
| Input 5     | `point_labels` — float32 [1, N]                 |
| Input 6     | `mask_input` — float32 [1, 1, 256, 256]         |
| Input 7     | `has_mask_input` — float32 [1]                   |
| Output 1    | `masks` — float32 [1, 3, 1024, 1024]            |
| Output 2    | `iou_predictions` — float32 [1, 3]              |
| Output 3    | `low_res_masks` — float32 [1, 3, 256, 256]      |

- `point_coords`: click (x, y) in 1024x1024 coordinates
- `point_labels`: 1 = foreground, 0 = background
- `mask_input`: previous iteration mask (zeros for first click)
- `has_mask_input`: 1.0 if mask_input is valid, 0.0 otherwise

## Notes

- Multi-mask output: 3 candidate masks per prompt, select by highest IoU score.
- Output masks are always 1024x1024 (resize to original image size at runtime).
- `low_res_masks` (256x256) can be fed back as `mask_input` for iterative refinement.
- Exported with FP32 precision.
- Conversion uses the shared convert.py from mask-object-sam21-small.

## Selection Criteria

| Property                 | Value                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                       |
| OSAID v1.0               | Open Weights                                                                                     |
| MOF                      | Class II (Open Tooling)                                                                          |
| Training data license    | SA-V: CC BY 4.0; SA-1B: custom Meta research-only license                                       |
| Training data provenance | [SA-V](https://ai.meta.com/datasets/segment-anything-video/) (50.9K videos) + [SA-1B](https://ai.meta.com/datasets/segment-anything/) (11M stock images) |
| Training code            | [Apache-2.0](https://github.com/facebookresearch/sam2)                                          |
| Known limitations        | SA-1B: unnamed stock provider, research-only license (not OSI), prohibits commercial use/redistribution, requires data destruction within 3 months |
| Published research       | [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) (2024)          |
| Inference                | Local only, no cloud dependencies                                                                |
| Scope                    | Interactive object segmentation                                                                  |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                      |
