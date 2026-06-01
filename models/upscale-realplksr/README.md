# RealPLKSR

Real-world variant of PLKSR (Partial Large Kernel CNN for Super-Resolution),
robust to typical photo artefacts (noise, blur, JPEG/WebP compression).
Lightweight pure-CNN architecture – faster than transformer-based upscalers.

Includes both 2x and 4x upscaling variants.

## Source

- Architecture and weights: [dslisleedh/PLKSR](https://github.com/dslisleedh/PLKSR) (MIT) – Dongheon Lee et al., Machine Intelligence Laboratory, University of Seoul
- Paper: [Partial Large Kernel CNNs for Efficient Super-Resolution](https://arxiv.org/abs/2404.11848) (2024)
- Checkpoints: dslisleedh's MSSIM pretrain release – [see issue #4](https://github.com/dslisleedh/PLKSR/issues/4) (MIT, same as architecture)
  - x2: `2x_realplksr_mssim_pretrain.pth`
  - x4: `4x_realplksr_mssim_pretrain.pth`
- Trained via the [neosr](https://github.com/neosr-project/neosr) framework with the RealESRGAN degradation pipeline
- Loaded for conversion via [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel), which auto-detects the RealPLKSR variant from the checkpoint state_dict

## Architecture

PLKSR replaces standard depthwise large-kernel convolutions with *partial*
large kernels – applied only to a subset of channels – reducing FLOPs while
keeping the receptive field that drives SR quality. The Real-world variant
swaps the upsampler for DySample and is trained with stronger augmentation
for robustness to typical photo artefacts (noise, blur, compression).

The shipped weights are the MSSIM-pretrain stage (no GAN finetune) –
faithful, conservative output without the texture hallucination risk that
GAN-trained SR models exhibit.

| Property     | Value                                                |
|--------------|------------------------------------------------------|
| Architecture | RealPLKSR                                            |
| Parameters   | ~7M                                                  |
| Receptive    | Large (partial 17×17 kernels)                        |
| Upsampler    | DySample                                             |
| Loss         | MSSIM (pretrain stage)                               |

## ONNX Models

| Property   | model_x2.onnx                        | model_x4.onnx                          |
|------------|--------------------------------------|----------------------------------------|
| Input      | `input` – float32 [1, 3, 512, 512]   | `input` – float32 [1, 3, 256, 256]     |
| Output     | `output` – float32 [1, 3, 1024, 1024]| `output` – float32 [1, 3, 1024, 1024]  |
| Resolution | Static, baked at 512×512             | Static, baked at 256×256               |
| Opset      | 20                                   | 20                                     |
| Normalize  | [0, 1] range (divide by 255)         | [0, 1] range (divide by 255)           |
| Tiling     | Yes (`model_x2.input_sizes: [512]`)  | Yes (`model_x4.input_sizes: [256]`)    |

Both variants produce a 1024×1024 output tile – x2 from a 512×512 input,
x4 from a 256×256 input. Per-stem tile sizes are declared in the manifest
so darktable picks the right size for each variant at runtime:

```yaml
attributes:
  model_x2:
    input_sizes: [512]
  model_x4:
    input_sizes: [256]
```

## Notes

- Input and output are RGB images in [0, 1] range.
- Output should be clipped to [0, 1] before converting back to uint8.
- Exported with FP32 precision. FP16 export is supported via `--fp16` in
  convert args but off by default.
- Inputs are baked into the graph so JIT-compiling EPs (CoreML,
  MIGraphX) only pay the compile cost once. Callers must tile at
  exactly the declared size.
- Conversion uses [Spandrel](https://github.com/chaiNNer-org/spandrel)
  to auto-detect the RealPLKSR variant from the checkpoint's state_dict,
  avoiding the need to clone PLKSR or neosr.

## Selection Criteria

| Property                 | Value                                                                                               |
|--------------------------|-----------------------------------------------------------------------------------------------------|
| Model license            | MIT (weights and architecture)                                                                      |
| OSAID v1.0               | Open Source AI                                                                                      |
| MOF                      | Class II (Open Tooling)                                                                             |
| Training data license    | DF2K (DIV2K + Flickr2K) per neosr common practice; Flickr2K without an explicit open-source license |
| Training data provenance | Synthetic real-world degradations applied via the neosr framework                                   |
| Training code            | [PLKSR](https://github.com/dslisleedh/PLKSR) (MIT) + [neosr](https://github.com/neosr-project/neosr) (Apache-2.0) |
| Known limitations        | MSSIM-pretrain checkpoints only (no GAN finetune) – conservative output, no hallucinated detail     |
| Published research       | [Partial Large Kernel CNNs for Efficient Super-Resolution](https://arxiv.org/abs/2404.11848)        |
| Inference                | Local only, no cloud dependencies                                                                   |
| Scope                    | Image upscaling (2x and 4x super-resolution, robust to noise/blur/compression artefacts)            |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                         |
