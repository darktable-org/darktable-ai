"""Click CLI entry point for darktable-ai."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click

from darktable_ai.config import ModelConfig, load_model_config
from darktable_ai.discovery import discover_models, find_project_root


def _get_root(ctx: click.Context) -> Path:
    return ctx.obj["root"]


def _load_config(root: Path, model_id: str) -> ModelConfig:
    model_dir = root / "models" / model_id
    if not model_dir.is_dir():
        click.echo(f"Error: Model '{model_id}' not found in models/", err=True)
        sys.exit(1)
    return load_model_config(model_dir, root)


def _sync_deps(config: ModelConfig) -> None:
    """Ensure the model's dependency group is installed."""
    group = config.dep_group
    if group == "core":
        return
    click.echo(f"  Syncing dependency group: {group}")
    subprocess.run(
        ["uv", "sync", "--group", group],
        cwd=str(config.root_dir),
        check=True,
    )


def _for_each_model(
    root: Path, model_id: str | None, callback, *, sync: bool = False
) -> None:
    """Run callback for one model or all non-skipped models."""
    if model_id:
        config = _load_config(root, model_id)
        if sync:
            _sync_deps(config)
        callback(config)
    else:
        for config in discover_models(root):
            if config.skip:
                click.echo(f"Skipping {config.id} (.skip)")
                continue
            click.echo(f"\n{'=' * 40}")
            click.echo(f"  {config.id}")
            click.echo(f"{'=' * 40}")
            if sync:
                _sync_deps(config)
            callback(config)


@click.group()
@click.version_option(package_name="darktable-ai")
@click.pass_context
def main(ctx):
    """darktable-ai: AI model pipeline for darktable"""
    ctx.ensure_object(dict)
    ctx.obj["root"] = find_project_root()


@main.command()
@click.argument("model_id", required=False)
@click.pass_context
def setup(ctx, model_id):
    """Download checkpoints and run repo setup commands."""
    from darktable_ai.download import download_checkpoints

    root = _get_root(ctx)

    def _setup(config: ModelConfig):
        if config.repo:
            repo_dir = config.repo_dir
            if repo_dir and not repo_dir.is_dir():
                click.echo(f"  Initializing submodule: {config.repo.submodule}")
                subprocess.run(
                    ["git", "submodule", "update", "--init", config.repo.submodule],
                    cwd=str(root), check=True,
                )

            if config.repo.setup and repo_dir and repo_dir.is_dir():
                click.echo(f"  Running repo setup: {config.repo.setup}")
                env = os.environ.copy()
                env["DTAI_ROOT"] = str(root)
                subprocess.run(
                    config.repo.setup, shell=True,
                    cwd=str(repo_dir), env=env, check=True,
                )

        if config.checkpoints:
            download_checkpoints(config.checkpoints, root)

    _for_each_model(root, model_id, _setup)


@main.command()
@click.argument("model_id", required=False)
@click.pass_context
def convert(ctx, model_id):
    """Convert model checkpoints to ONNX."""
    from darktable_ai.convert import run_conversion

    root = _get_root(ctx)
    _for_each_model(root, model_id, run_conversion, sync=True)


@main.command()
@click.argument("model_id", required=False)
@click.pass_context
def validate(ctx, model_id):
    """Validate ONNX model output."""
    from darktable_ai.validate import run_validation

    root = _get_root(ctx)
    _for_each_model(root, model_id, run_validation, sync=True)


@main.command("package")
@click.argument("model_id", required=False)
@click.pass_context
def package_cmd(ctx, model_id):
    """Package model as .dtmodel archive."""
    from darktable_ai.package import package_model

    root = _get_root(ctx)
    _for_each_model(root, model_id, package_model)


@main.command()
@click.argument("model_id", required=False)
@click.pass_context
def demo(ctx, model_id):
    """Run demo inference on sample images."""
    from darktable_ai.demo import run_demo

    root = _get_root(ctx)
    _for_each_model(root, model_id, run_demo, sync=True)


@main.command()
@click.argument("model_id", required=False)
@click.pass_context
def run(ctx, model_id):
    """Run full pipeline: setup -> convert -> validate -> package -> demo."""
    from darktable_ai.convert import run_conversion
    from darktable_ai.demo import run_demo
    from darktable_ai.download import download_checkpoints
    from darktable_ai.package import package_model
    from darktable_ai.validate import run_validation

    root = _get_root(ctx)

    def _run_pipeline(config: ModelConfig):
        click.echo("\n=== Setup ===")
        ctx.invoke(setup, model_id=config.id)

        click.echo("\n=== Convert ===")
        run_conversion(config)

        click.echo("\n=== Validate ===")
        run_validation(config)

        click.echo("\n=== Package ===")
        package_model(config)

        click.echo("\n=== Demo ===")
        run_demo(config)

    _for_each_model(root, model_id, _run_pipeline, sync=True)


@main.command("list")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON for CI")
@click.pass_context
def list_models(ctx, as_json):
    """List available models."""
    root = _get_root(ctx)
    models = discover_models(root)

    if as_json:
        matrix = [
            {"id": m.id, "dep_group": m.dep_group}
            for m in models if not m.skip
        ]
        click.echo(json.dumps(matrix))
    else:
        for config in models:
            status = " (skipped)" if config.skip else ""
            click.echo(f"  {config.id:<35} {config.task:<15} {config.description}{status}")


@main.command("versions")
@click.pass_context
def versions(ctx):
    """Generate versions.json from model.yaml files."""
    root = _get_root(ctx)
    models = discover_models(root)
    data = {
        "models": {
            m.id: m.version
            for m in sorted(models, key=lambda m: m.id)
            if not m.skip
        }
    }
    output_path = root / "output" / "versions.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    click.echo(f"Generated {output_path}")


@main.command("eval")
@click.argument("task")
@click.argument("model_id")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def evaluate(ctx, task, model_id, extra_args):
    """Evaluate MODEL_ID on TASK benchmark."""
    from darktable_ai.evaluate import run_evaluation

    root = _get_root(ctx)
    run_evaluation(task, model_id, root, extra_args)
