"""Run model conversion steps and generate config.json."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from darktable_ai.config import ModelConfig


def run_conversion(config: ModelConfig) -> None:
    """Execute all conversion steps defined in model.yaml."""
    if not config.convert:
        print("  No conversion steps defined — skipping.")
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_vendor_paths(config)

    for i, step in enumerate(config.convert, 1):
        label = f"[{i}/{len(config.convert)}]" if len(config.convert) > 1 else ""
        print(f"  {label} {step.script}".strip())

        module = _import_script(config.model_dir / step.script)
        kwargs = _resolve_args(config, step.args)
        module.convert(**kwargs)

    generate_config_json(config)


def generate_config_json(config: ModelConfig) -> None:
    """Write config.json to the model's output directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_file = config.output_dir / "config.json"

    data = {
        "id": config.id,
        "name": config.name,
        "description": config.description,
        "task": config.task,
        "arch": config.arch,
        "backend": "onnx",
        "version": "1.0",
        "tiling": config.tiling,
    }

    config_file.write_text(json.dumps(data, indent=4) + "\n")
    print(f"  Generated: {config_file}")


def _import_script(script_path: Path):
    """Import a Python script as a module."""
    spec = importlib.util.spec_from_file_location(
        script_path.stem, script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_args(config: ModelConfig, args: dict) -> dict:
    """Resolve template variables in step args."""
    resolved = {}
    for key, value in args.items():
        if isinstance(value, str):
            resolved[key] = config.resolve_template(value)
        else:
            resolved[key] = value
    return resolved


def _setup_vendor_paths(config: ModelConfig) -> None:
    """Add vendor submodule paths to sys.path for model imports."""
    if not config.repo:
        return

    repo_dir = config.repo_dir
    if not repo_dir or not repo_dir.is_dir():
        return

    # Common patterns: repo itself, repo/src, repo/<name>
    for subdir in [repo_dir, repo_dir / "src"]:
        path = str(subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
