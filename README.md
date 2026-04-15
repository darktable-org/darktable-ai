# Darktable AI Models

AI model conversion and packaging pipeline for [darktable](https://www.darktable.org/) – an open-source photography workflow application and raw developer ([GitHub](https://github.com/darktable-org/darktable)).

Currently targets the ONNX backend. The pipeline is designed to support additional backends as darktable gains support for other AI runtimes.

## Models

| Model                                                                         | Task    | Description                                     |
|-------------------------------------------------------------------------------|---------|-------------------------------------------------|
| [`denoise-nafnet`](models/denoise-nafnet/README.md)                           | denoise | NAFNet denoiser trained on SIDD dataset         |
| [`denoise-nind`](models/denoise-nind/README.md)                               | denoise | UNet denoiser trained on NIND dataset           |
| [`embed-openclip-vitb32`](models/embed-openclip-vitb32/README.md)             | embed   | OpenCLIP ViT-B/32 text/image embeddings         |
| [`mask-object-sam21-base-plus`](models/mask-object-sam21-base-plus/README.md) | mask    | SAM 2.1 Hiera Base Plus for interactive masking |
| [`mask-object-sam21-small`](models/mask-object-sam21-small/README.md)         | mask    | SAM 2.1 Hiera Small for interactive masking     |
| [`mask-object-sam21-tiny`](models/mask-object-sam21-tiny/README.md)           | mask    | SAM 2.1 Hiera Tiny for interactive masking      |
| [`mask-object-segnext-b2hq`](models/mask-object-segnext-b2hq/README.md)      | mask    | SegNext ViT-B SAx2 HQ for semantic masking      |
| [`upscale-bsrgan`](models/upscale-bsrgan/README.md)                          | upscale | BSRGAN 2x and 4x blind super-resolution        |

## Repository structure

```
pyproject.toml        Project configuration, dependency groups, CLI entry point
darktable_ai/         Python package (CLI + pipeline orchestration)
vendor/               Git submodules (nind-denoise, sam2, SegNext)
samples/<task>/        Sample images organized by task
output/               Build output: ONNX models + config.json (gitignored)
temp/                 Downloaded checkpoints (gitignored)
models/
  <model>/
    model.yaml        Model metadata, checkpoints, conversion steps
    convert.py        Model-specific conversion script
    demo.py           Demo inference script
    .skip             If present, skip this model in batch operations and CI
```

## Requirements

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.12.

Dependencies are managed through [dependency groups](https://docs.astral.sh/uv/concepts/dependency-groups/) in `pyproject.toml`. The base package only needs `click` and `pyyaml`. ML dependencies are split into groups – one per model plus a shared `core` group – so you only install what you need. Use `uv sync --group <name>` to install a specific group, or `--group all-models` for everything.

## Setup

```bash
git clone --recurse-submodules https://github.com/<org>/darktable-ai.git
cd darktable-ai

# Install CLI + core ML dependencies
uv sync --group core

# Or install deps for a specific model
uv sync --group nind

# Or install everything
uv sync --group all-models
```

## Usage

```bash
# List available models
uv run dtai list

# Run full pipeline for a single model
uv run dtai run denoise-nind

# Run full pipeline for all models
uv run dtai run

# Run individual steps
uv run dtai setup denoise-nind       # Download checkpoints
uv run dtai convert denoise-nind     # Convert to ONNX + generate config.json
uv run dtai validate denoise-nind    # Validate ONNX output
uv run dtai package denoise-nind     # Create .dtmodel archive
uv run dtai demo denoise-nind        # Run demo on sample images

# Generate versions.json
uv run dtai versions

# Evaluate a model
uv sync --group eval
uv run dtai eval mask mask-object-segnext-b2hq --limit 5
```

## Versioning

Each model has a `version` field in its `model.yaml` that gets written to `config.json`. The `dtai versions` command generates `output/versions.json` – a manifest mapping model IDs to their current versions:

```json
{
  "models": {
    "denoise-nind": "1.0",
    "mask-object-sam21-small": "1.0"
  }
}
```

This file is published as a GitHub release asset alongside `.dtmodel` packages. darktable fetches it to check for model updates without downloading full packages.

## Demos

Each model includes a `demo.py` script that runs inference on sample images
from `samples/<task>/`. Models that require per-image input (e.g. point prompts
for object segmentation) define `image_args` in their `model.yaml`.

Output images are saved to `output/<model>-demo/`.

## Model selection criteria

Darktable is free software licensed under [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html). All AI models included in this repository are selected with the following principles in mind.

### Open source compliance

Each model card documents the following and must meet the stated requirements:

- **GPL-3.0-compatible license.** Model weights must be released under a license compatible with GPL-3.0 (e.g. Apache-2.0, MIT, BSD, GPL-3.0). Proprietary or non-commercial-only models are not accepted.
- **[OSAID v1.0](https://opensource.org/ai/open-source-ai-definition) classification.** Open Source AI, Open Weights, or Open Model.
- **[MOF](https://isitopen.ai/) classification.** Class I (Open Science), Class II (Open Tooling), or Class III (Open Model).
- **Training data license.** Specific license(s) for each training dataset.
- **Training data provenance.** Where data came from and how it was collected. Models trained on undisclosed or scraped personal data without consent are not accepted.
- **Training code availability.** Link to public training code under an open-source license.
- **Known limitations.** What cannot be audited or verified (e.g. non-releasable pre-training data, non-OSI training data licenses).

### Published research

- **Peer-reviewed or public report.** Models should have an accompanying peer-reviewed paper or public technical report describing the architecture and training methodology.

### Responsible use

- **Privacy by design.** All inference runs locally on the user's machine. No data is sent to external services. No telemetry, no cloud dependencies.
- **Purpose-limited scope.** Models are selected for photo editing tasks: denoising, masking, depth estimation, and object removal (inpainting), etc. We do not include models designed for generating, manipulating, or synthesizing human likenesses.
- **Reproducibility.** Conversion scripts, model configurations, and source references are fully documented so that any user can verify and rebuild the ONNX models from the original checkpoints.

## Adding a new model

1. Create `models/<model>/model.yaml` with model metadata, checkpoint URLs, and conversion steps
2. Create `models/<model>/convert.py` with model-specific conversion logic
3. Create `models/<model>/demo.py` with inference script
4. Create `models/<model>/README.md` with the model card (see below)
5. Add sample images to `samples/<task>/`
6. If the model depends on an external repo, add it as a git submodule under `vendor/`
7. Add a dependency group to `pyproject.toml` if the model needs extra packages
8. Run `uv run dtai run <model>` to build and test

### convert.py

The conversion script must expose a `convert()` function that the pipeline calls directly. It receives keyword arguments matching the `args` dict in `model.yaml` (with template variables resolved). Keep `main()` with argparse for standalone use.

```python
def convert(checkpoint, output, opset=17, fp16=False):
    """Entry point called by the pipeline."""
    model = load_model(checkpoint)
    export_to_onnx(model, output, opset_version=opset, fp16=fp16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.opset, args.fp16)

if __name__ == "__main__":
    main()
```

The corresponding `model.yaml` args use Python keyword names (not CLI flags):

```yaml
convert:
  - script: convert.py
    args:
      checkpoint: "{temp}/model.pth"
      output: "{output}/model.onnx"
      opset: 17
      fp16: true
```

Available template variables: `{root}`, `{model_dir}`, `{temp}`, `{output}`, `{repo}`.

### demo.py

The demo script must expose a `demo()` function. The first arguments depend on the model type:

- **single** (`type: single`): `demo(model, image, output, **kwargs)`
- **split** (`type: split`): `demo(encoder, decoder, image, output, **kwargs)`
- **multi** (`type: multi`): `demo(model_dir, image, output, **kwargs)`

The pipeline passes `image` and `output` paths automatically. Any per-image arguments defined in `model.yaml` under `demo.image_args` are passed as extra `**kwargs`.

```python
def demo(model, image, output, **kwargs):
    """Entry point called by the pipeline."""
    run_inference(model, image, output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    demo(args.model, args.image, args.output)

if __name__ == "__main__":
    main()
```

### Model card (README.md)

Each model directory must include a `README.md` documenting:

- **Source** – repository URL, paper reference, license
- **Architecture** – brief description of the model architecture
- **ONNX Models** – input/output tensor names, shapes, data types, normalization, tiling support
- **Selection Criteria** – a table covering all items from the [model selection criteria](#model-selection-criteria):

| Property                 | Value              |
|--------------------------|--------------------|
| Model license            | (e.g. Apache-2.0)  |
| OSAID v1.0               | (e.g. Open Source AI) |
| MOF                      | (e.g. Class II)    |
| Training data license    | ...                |
| Training data provenance | ...                |
| Training code            | (link)             |
| Known limitations        | ...                |
| Published research       | (link to paper)    |
| Inference                | Local only, no cloud dependencies |
| Scope                    | (e.g. Image denoising) |
| Reproducibility          | Full pipeline      |

See existing model READMEs for examples.
