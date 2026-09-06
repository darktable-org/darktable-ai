"""Run demo inference on sample images."""

from __future__ import annotations

import json
from pathlib import Path

from darktable_ai.config import ModelConfig
from darktable_ai.convert import _import_script

_PROCESSED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_RAW_IMAGE_EXTS = {
    ".cr2", ".cr3", ".crw",      # Canon
    ".nef", ".nrw",              # Nikon
    ".arw", ".sr2", ".srf",      # Sony
    ".raf",                      # Fuji
    ".rw2",                      # Panasonic
    ".pef", ".ptx",              # Pentax
    ".orf",                      # Olympus
    ".rwl",                      # Leica
    ".srw",                      # Samsung
    ".dng",                      # Adobe generic
}
_SAMPLE_EXTS = _PROCESSED_IMAGE_EXTS | _RAW_IMAGE_EXTS

# Task → output file extension. Raw-domain tasks can't round-trip through PNG
# because they produce linear HDR or >8-bit data; embedding models produce no
# image at all.
_OUTPUT_EXT_BY_TASK = {
    "rawdenoise": ".tif",
    "embedding": ".json",
}


def run_demo(config: ModelConfig) -> None:
    """Run the model's demo.py on all sample images for its task."""
    images_dir = config.root_dir / "samples" / config.task
    if not images_dir.is_dir():
        print(f"  No samples directory found: {images_dir}")
        return

    demo_script = config.model_dir / "demo.py"
    if not demo_script.is_file():
        print(f"  No demo.py found in {config.model_dir}")
        return

    demo_output_dir = config.root_dir / "output" / f"{config.id}-demo"
    demo_output_dir.mkdir(parents=True, exist_ok=True)

    module = _import_script(demo_script)
    model_kwargs = _model_type_kwargs(config)
    out_ext = _OUTPUT_EXT_BY_TASK.get(config.task, ".png")

    samples = sorted(p for p in images_dir.rglob("*")
                     if p.is_file() and p.suffix.lower() in _SAMPLE_EXTS)

    for img in samples:
        if img.stem.startswith("expected"):
            continue

        rel = img.relative_to(images_dir).with_suffix("")
        name = str(rel).replace("/", "_").replace("\\", "_")
        output_path = demo_output_dir / f"{name}{out_ext}"
        extra_kwargs = _image_kwargs(config, img, rel)

        print(f"  {name}")
        module.demo(
            **model_kwargs,
            image=str(img),
            output=str(output_path),
            **extra_kwargs,
        )


def _model_type_kwargs(config: ModelConfig) -> dict:
    """Build model path kwargs for demo() based on model type."""
    output_dir = config.output_dir
    if config.type == "split":
        return {
            "encoder": str(output_dir / "encoder.onnx"),
            "decoder": str(output_dir / "decoder.onnx"),
        }
    elif config.type == "multi":
        return {"model_dir": str(output_dir)}
    else:
        return {"model": str(output_dir / "model.onnx")}


def _image_kwargs(config: ModelConfig, img: Path, rel: Path) -> dict:
    """Get extra demo kwargs for a specific image.

    Reads from a JSON sidecar file next to the sample image first
    (e.g. ``samples/mask-object/example_01.json``), then falls back
    to ``demo.image_args`` in model.yaml, keyed by either the flattened
    relative path or the bare filename stem.
    """
    sidecar = img.with_suffix(".json")
    if sidecar.is_file():
        with open(sidecar) as f:
            return json.load(f)
    flat = str(rel).replace("/", "_").replace("\\", "_")
    return (config.demo.image_args.get(flat)
            or config.demo.image_args.get(img.stem, {}))
