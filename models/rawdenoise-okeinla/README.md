# Okeinla 1.0

Okeinla jointly denoises and demosaics Bayer RAW images. It uses NAFNet with
a packed-Bayer input and a pixel-shuffle camera-RGB output.

## Source

- [Training code](https://github.com/AyedaOk/okeinla), pinned by `vendor/okeinla`.
- [Checkpoint](https://huggingface.co/AyedaOk/okeinla/tree/7d4480d305ac9a425832afb9c110ee0b1af0200a).
- [Training dataset](https://huggingface.co/datasets/AyedaOk/okeinla-dataset).
- Credits: [RawNIND](https://github.com/trougnouf/rawnind_jddc),
  [NAFNet](https://github.com/megvii-research/NAFNet), and the
  [RawForge/Restorer loss recipe](https://github.com/rymuelle/Restorer).

## ONNX model

`model_bayer.onnx` takes `input`, float32 `[1, 4, 512, 512]`, and returns
`output`, float32 `[1, 3, 1024, 1024]` camera RGB.
Normalize black/white levels and pack RGGB channels without input white balance.
Use cropped mirror padding, tiled inference, and output gain matching. Apply
camera colour conversion after inference. The graph takes no ISO input.

## Selection criteria

| Property | Value |
| --- | --- |
| Model license | GPL-3.0-only; NAFNet components retain their MIT notice |
| OSAID v1.0 | Open Weights; no claim of full OSAID compliance |
| MOF | Unclassified: public denoising evaluation results are missing from the release |
| Training data license | CC BY-SA 4.0 for both datasets; [RawNIND license record](https://doi.org/10.14428/DVN/DEQCIM) |
| Training data provenance | RawNIND and Okeinla dataset: paired noisy and reference RAW photographs. Okeinla captures use a tripod-mounted Sony A7C II with base-ISO references. |
| Training code | Public GPLv3 source linked above |
| Known limitations | Bayer only; no X-Trans or linear-RGB variant; no bitwise retraining claim |
| Published research | [Okeinla technical report](#technical-report); upstream [RawNIND](https://arxiv.org/abs/2501.08924) and [NAFNet](https://arxiv.org/abs/2204.04676) papers |
| Inference | Local; no telemetry or cloud inference |
| Scope | RAW denoising and demosaicing |
| Reproducibility | Pinned source and checkpoint; converter checks checkpoint SHA256 and PyTorch/ONNX parity |

## Build

```sh
uv run dtai run rawdenoise-okeinla
```

The demo reuses RawNIND's Bayer pipeline and skips non-Bayer samples.
The package includes GPLv3 and NAFNet MIT license texts.

## Technical report

### Architecture and training

Okeinla 1.0 uses a width-32 NAFNet with 29,162,604 parameters. The encoder has
four stages with 2, 2, 4, and 8 blocks; the middle has 12 blocks; each decoder
stage has 2 blocks. A pixel-shuffle head produces camera RGB at sensor resolution.
See the pinned [architecture source](../../vendor/okeinla/src/rawnind/models/nafnet.py).

Training uses paired RAW photographs from RawNIND and the Okeinla dataset.
The Okeinla dataset contains 106 Sony A7C II scenes, 761 noisy exposures, and
368 base-ISO references. The configurations define scene-level validation and
test reserves, 256-pixel training crops, four crops per image, batch size four,
and output gain matching. They use seed 2026 with deterministic mode disabled.

The loss combines shadow-weighted L1 (0.16), MS-SSIM loss (0.84), and frozen
ImageNet VGG16 feature L1 loss. The final stages increase the feature-loss
weight from 0.025 to 0.035. VGG16 serves only as a training loss network;
inference needs no VGG16 weights. See the pinned
[loss implementation](../../vendor/okeinla/src/common/libs/pt_losses.py).

The released lineage runs from M5B checkpoint 2915000 to M8 checkpoint 2925000,
then continuation checkpoint 2940000. M8 resets Adam at learning rate `1e-5`;
the continuation retains its optimizer state. The published
[configurations](../../vendor/okeinla/src/rawnind/config/) record these stages.
Checkpoint labels do not establish the number of training updates, and the
release does not claim an exact replay of historical training.

### Evaluation and limits

The converter checks PyTorch/ONNX agreement on a seeded 512-pixel packed-Bayer
tile with `rtol=1e-3` and `atol=1e-5`. This checks export correctness, not denoising
quality. This release provides no public quantitative denoising evaluation
results, so this card makes no quality-ranking or cross-camera performance claim.

Local checks passed on Python 3.12 with the pinned checkpoint: strict weight
loading, FP32 ONNX export, PyTorch/ONNX agreement, and package validation.
Four unit tests cover Bayer layouts, invalid mosaics, checkpoint rejection,
and unsupported sensors. PR CI runs these tests before conversion.
The full-image demo processed `1D3_6400.CR2` and skipped an X-Trans RAF.
These smoke tests do not measure denoising quality.

The [MOF component requirements](https://github.com/lfai/model_openness_tool/blob/main/web/modules/mof/config/install/mof.settings.yml)
include evaluation results even for Class III. The release cannot claim that
class from weights and source availability alone.
