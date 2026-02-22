# Darktable AI Models

ONNX models and conversion scripts for [darktable](https://www.darktable.org/) - an open-source photography workflow application and raw developer ([GitHub](https://github.com/darktable-org/darktable)).

## Models

| Model                                                                           | Task    | Description                                   |
|---------------------------------------------------------------------------------|---------|-----------------------------------------------|
| [`denoise-nafnet`](models/denoise-nafnet/README.md)                             | denoise | NAFNet denoiser trained on SIDD dataset       |
| [`denoise-nind`](models/denoise-nind/README.md)                                 | denoise | UNet denoiser trained on NIND dataset         |
| [`mask-object-segnext-b2hq`](models/mask-object-segnext-b2hq/README.md)         | mask    | SegNext ViT-B SAx2 HQ for masking             |

## Repository structure

```
run.sh                Run pipeline for a model (supports subcommands)
run_all.sh            Run full pipeline for all models
images/               Sample images for demos
output/               Build output: ONNX models + generated config.json (gitignored)
temp/                 Downloaded checkpoints before conversion (gitignored)
models/
  common.sh           Shared shell functions (setup, convert, validate, demo, clean)
  validate.py         Validate ONNX output (load models, check config, print I/O shapes)
  <model>/
    model.conf        Model metadata, configuration, conversion, and demo functions
    requirements.txt  Python dependencies
    convert.py        Model-specific conversion script
    demo.py           Demo inference script
    README.md         Model documentation and ONNX tensor specs
    .skip             If present, skip this model in run_all.sh and CI
```

## Usage

Run the full pipeline (setup, convert, validate, demo, clean) for a single model:

```bash
./run.sh <model_id>
```

Run the pipeline for all models:

```bash
./run_all.sh
```

Or run each step individually:

```bash
./run.sh <model_id> setup     # Create venv, clone repo, download checkpoint
./run.sh <model_id> convert   # Convert to ONNX + generate config.json
./run.sh <model_id> validate  # Load ONNX, verify config, print I/O shapes
./run.sh <model_id> package   # Create .dtmodel package (zip archive)
./run.sh <model_id> demo      # Run demo on sample images
./run.sh <model_id> clean     # Remove venv and cloned repo
```

## Demos

Each model includes a `demo.py` script that runs inference on the sample images
in `images/`. Models that require per-image input (e.g. point prompts for object
segmentation) define a `demo_args()` function in their `model.conf`.

Output images are saved to `models/<model>/output/`.

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

1. Create `models/<model>/model.conf` with model metadata (`MODEL_ID`, `MODEL_NAME`, `MODEL_DESCRIPTION`, `MODEL_TASK`), repo/checkpoint URLs, `run_convert()`, and optional `demo_args()`
2. Create `models/<model>/requirements.txt` with Python dependencies (must include `onnxruntime`)
3. Create `models/<model>/convert.py` with model-specific conversion logic
4. Create `models/<model>/demo.py` with inference script
5. Run `./run.sh <model> convert` to build ONNX output and generate `config.json`
