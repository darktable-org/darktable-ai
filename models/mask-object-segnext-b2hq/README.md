# SegNext ViT-B SAx2 HQ

SegNext interactive segmentation model — ViT-B backbone with SAx2 cross-attention,
fine-tuned on HQSeg-44K for high-quality mask boundaries.

## Source

- Repository: https://github.com/uncbiag/SegNext
- Paper: [SegNext: Rethinking Interactive Image Segmentation with Low Latency, High Quality, and Diverse Prompts](https://arxiv.org/abs/2312.01171) (CVPR 2024)
- License: MIT

## Architecture

ViT-B backbone (MAE pre-trained) with sparse windowed attention, self-attention
prompt fusion (SAx2), SimpleFPN neck, and lightweight segmentation head. Prompts
are encoded as distance maps from click coordinates.

## ONNX Models

Split into encoder (image features) and decoder (prompt fusion + mask prediction).

### encoder.onnx (~339 MB)

| Direction | Tensor | Shape | Type |
| --- | --- | --- | --- |
| in | image | 1 x 3 x 1024 x 1024 | float32 |
| out | image_feats | 1 x 768 x 64 x 64 | float32 |

### decoder.onnx (~103 MB)

| Direction | Tensor | Shape | Type |
| --- | --- | --- | --- |
| in | image_feats | 1 x 768 x 64 x 64 | float32 |
| in | point_coords | 1 x N x 2 | float32 |
| in | point_labels | 1 x N | float32 |
| in | prev_mask | 1 x 1 x 1024 x 1024 | float32 |
| out | mask | 1 x 1 x 1024 x 1024 | float32 |

- `point_coords`: click (x, y) in 1024x1024 coordinates
- `point_labels`: 1 = foreground, 0 = background, -1 = padding
- `prev_mask`: previous iteration mask (zeros for first click)
- Distance maps are computed inside the ONNX decoder from raw coordinates

## Selection Criteria

| Property                 | Value                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------|
| Model license            | MIT                                                                                              |
| OSAID v1.0               | Open Source AI                                                                                   |
| MOF                      | Class I (Open Science)                                                                           |
| Training data license    | COCO: CC BY 4.0; LVIS: CC BY 4.0; HQSeg-44K: mixed (see datasets)                              |
| Training data provenance | [COCO](https://cocodataset.org/) (118K images) + [LVIS](https://www.lvisdataset.org/) (100K images) + [HQSeg-44K](https://github.com/SysCV/sam-hq) (44K images, fine-tune) |
| Training code            | [MIT](https://github.com/uncbiag/SegNext)                                                       |
| Known limitations        | HQSeg-44K aggregates multiple datasets with varying licenses; individual dataset terms apply     |
| Published research       | [SegNext](https://arxiv.org/abs/2312.01171) (CVPR 2024)                                         |
| Inference                | Local only, no cloud dependencies                                                                |
| Scope                    | Interactive object segmentation                                                                  |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                      |
