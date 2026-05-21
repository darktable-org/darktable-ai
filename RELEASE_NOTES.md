We're pleased to announce the first release of the darktable AI models, 5.6.0!

The github release is here: [https://github.com/darktable-org/darktable-ai/releases/tag/release-5.6.0](https://github.com/darktable-org/darktable-ai/releases/tag/release-5.6.0).

These models are intended for darktable 5.6.0 and later. They power the AI features introduced in this darktable release – interactive object masking and neural restore (raw denoise, image denoise, upscale). Models are installed and managed from the AI tab in darktable's preferences.

## Models

### Object masking – used in the darkroom mask manager's AI object tool

- **mask sam2.1 hiera small** (`mask-object-sam21-small`) – Segment Anything 2.1 (Hiera Small). Click on an object to generate a precise mask; click again with foreground/background prompt points to refine. Encoder + decoder pair: the encoder runs once per image (with optional GPU acceleration), the lightweight decoder produces masks interactively. Default choice for most users.

- **mask sam2.1 hiera tiny / base plus** (`mask-object-sam21-tiny`, `mask-object-sam21-base-plus`) – lighter and heavier variants of the same SAM 2.1 pipeline. Use *tiny* on low-memory systems or for faster encoder runs; use *base plus* when *small* doesn't capture an object cleanly.

- **mask segnext vitb-sax2 hq** (`mask-object-segnext-b2hq`) – SegNext ViT-B SAx2 HQ fine-tuned for interactive masking. Alternative to SAM 2.1 with openly documented training data. Also useful when SAM 2.1's segmentation behaviour doesn't fit a particular scene.

### Neural restore – used in the neural restore module in the lighttable/darkroom sidebar

- **denoise nind** (`denoise-nind`) – UNet denoiser from the NIND (Natural Image Noise Dataset) project, trained on Wikimedia Commons noisy/clean pairs. Drives the module's *denoise* task on demosaiced RGB; the result is written as a TIFF with the output ICC profile embedded and grouped with the source in the library.

- **denoise nafnet small** (`denoise-nafnet`) – lightweight NAFNet denoiser trained on the SIDD smartphone dataset. Alternative *denoise* task model – tuned for noise patterns typical of small-sensor cameras.

- **raw denoise nind** (`rawdenoise-nind`) – UtNet2 raw-domain denoiser trained on RawNIND. Drives the module's *raw denoise* task; the result is written as a DNG that re-enters the user's existing edit (Bayer or linear Rec.2020). The X-Trans variant falls back to the linear pipeline for now.

- **upscale bsrgan** (`upscale-bsrgan`) – BSRGAN blind super-resolution model with 2× and 4× tile sizes packaged together. Drives the module's *upscale* task; result is written as a TIFF embedding the output ICC profile.

## Compatibility

- Darktable version: 5.6.0.
- All models are statically-shaped ONNX with tile sizes declared in the manifest. Tiles are sized for both CPU inference and the GPU execution providers supported by darktable (CUDA, ROCm/MIGraphX, DirectML, OpenVINO, CoreML).