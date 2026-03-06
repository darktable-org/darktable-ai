# OpenCLIP ViT-B-32

Image embedding model for zero-shot tagging and perceptual similarity.
Uses contrastive language-image pre-training to produce 512-dimensional
embeddings that can be compared via cosine similarity.

## Source

- Repository: https://github.com/mlfoundations/open_clip
- Paper: [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143) (2023)
- Pretrained: `laion2b_s34b_b79k` (LAION-2B dataset)
- License: MIT

## Architecture

ViT-B-32 (Vision Transformer Base, 32x32 patches) — 12 transformer layers,
768 hidden dimension, 12 attention heads. Input image is split into 7x7 = 49
patches of 32x32 pixels plus a CLS token. The CLS token output is projected
to 512 dimensions.

## ONNX Model

| Property    | Value                                   |
|-------------|-----------------------------------------|
| File        | `model.onnx`                            |
| Input       | `image` — float32 [1, 3, 224, 224]     |
| Output      | `embedding` — float32 [1, 512]         |
| Resolution  | Fixed 224x224 (resize + center crop)    |
| Normalize   | [0, 1] range (CLIP norm baked in)       |

## Notes

- Input and output are both float32.
- Embeddings are L2-normalized — use dot product for cosine similarity.
- No tiling: the model requires fixed 224x224 input.
- Architecture is loaded via `open_clip` pip package (no repo clone needed).
- No checkpoints to download — `open_clip` fetches pretrained weights automatically during conversion.
- Output `tags.json` contains pre-computed text embeddings for ~80 photo tags.

## Selection Criteria

| Property                 | Value                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------|
| Model license            | MIT                                                                                                     |
| OSAID v1.0               | Open Source AI                                                                                          |
| MOF                      | Class II (Open Tooling)                                                                                 |
| Training data license    | LAION-2B (CC-BY-4.0 metadata); images are web-crawled with mixed licenses                               |
| Training data provenance | LAION-2B: 2B image-text pairs from Common Crawl, filtered using CLIP for quality                        |
| Training code            | [MIT](https://github.com/mlfoundations/open_clip)                                                      |
| Known limitations        | Training images are web-crawled; individual image licenses are not verified                              |
| Published research       | [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)   |
| Inference                | Local only, no cloud dependencies                                                                       |
| Scope                    | Image embedding for tagging and similarity (no generation or synthesis)                                  |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                             |
