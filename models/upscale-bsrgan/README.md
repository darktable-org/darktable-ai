# BSRGAN

Blind image super-resolution using practical degradation model.
Includes both 2x and 4x upscaling variants.

## Source

- Repository: https://github.com/cszn/BSRGAN
- Paper: [Designing a Practical Degradation Model for Deep Blind Image Super-Resolution](https://arxiv.org/abs/2103.14006) (ICCV 2021)
- License: Apache-2.0

## Architecture

RRDBNet (Residual-in-Residual Dense Block Network) — 23 RRDB blocks,
64 base features, 32 growth channels. Each RRDB block contains 3 dense blocks
with 5 convolutional layers each. Upsampling via nearest-neighbor interpolation
followed by convolution (1 step for 2x, 2 steps for 4x).

## ONNX Models

| Property    | model_x2.onnx                       | model_x4.onnx                       |
|-------------|--------------------------------------|--------------------------------------|
| Input       | `input` — float32 [1, 3, H, W]      | `input` — float32 [1, 3, H, W]      |
| Output      | `output` — float32 [1, 3, 2H, 2W]   | `output` — float32 [1, 3, 4H, 4W]   |
| Resolution  | Dynamic (any H, W)                   | Dynamic (any H, W)                   |
| Normalize   | [0, 1] range (divide by 255)         | [0, 1] range (divide by 255)         |
| Tiling      | Yes                                  | Yes                                  |

## Notes

- Input and output are both RGB images in [0, 1] range.
- Output should be clipped to [0, 1] before converting back to uint8.
- Exported with FP32 precision.
- Architecture is inlined in convert.py (no repo clone needed).

## Selection Criteria

| Property                 | Value                                                                                              |
|--------------------------|----------------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                         |
| OSAID v1.0               | Open Source AI                                                                                     |
| MOF                      | Class II (Open Tooling)                                                                            |
| Training data license    | DIV2K (CC0), Flickr2K, WED, OST — standard SR research datasets                                   |
| Training data provenance | Public image restoration benchmarks with synthetic practical degradation (blur, noise, JPEG, resize) |
| Training code            | [Apache-2.0](https://github.com/cszn/BSRGAN)                                                      |
| Known limitations        | Training datasets Flickr2K/WED/OST do not have explicit open-source licenses                       |
| Published research       | [Designing a Practical Degradation Model for Deep Blind Image Super-Resolution](https://arxiv.org/abs/2103.14006) (ICCV 2021) |
| Inference                | Local only, no cloud dependencies                                                                   |
| Scope                    | Image upscaling (2x and 4x blind super-resolution)                                                 |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                         |
