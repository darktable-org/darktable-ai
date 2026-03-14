"""Load and validate model.yaml configuration files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RepoConfig:
    submodule: str
    setup: str | None = None


@dataclass
class Checkpoint:
    url: str
    path: str


@dataclass
class ConvertStep:
    script: str
    args: dict[str, str | bool | int | float]


@dataclass
class DemoConfig:
    image_args: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    id: str
    name: str
    description: str
    task: str
    type: str = "single"
    arch: str = "generic"
    tiling: bool = False
    dep_group: str = "core"
    skip: bool = False

    repo: RepoConfig | None = None
    checkpoints: list[Checkpoint] = field(default_factory=list)
    convert: list[ConvertStep] = field(default_factory=list)
    demo: DemoConfig = field(default_factory=DemoConfig)

    model_dir: Path = field(default=Path("."), repr=False)
    root_dir: Path = field(default=Path("."), repr=False)

    @property
    def temp_dir(self) -> Path:
        return self.root_dir / "temp" / self.id

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "output" / self.id

    @property
    def repo_dir(self) -> Path | None:
        if self.repo:
            return self.root_dir / self.repo.submodule
        return None

    def resolve_template(self, value: str) -> str:
        """Replace template variables in a string."""
        return value.format(
            root=self.root_dir,
            model_dir=self.model_dir,
            temp=self.temp_dir,
            output=self.output_dir,
            repo=self.repo_dir or "",
        )


def load_model_config(model_dir: Path, root_dir: Path) -> ModelConfig:
    """Load a model.yaml file and return a ModelConfig."""
    yaml_path = model_dir / "model.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    repo = None
    if "repo" in data:
        repo = RepoConfig(
            submodule=data["repo"]["submodule"],
            setup=data["repo"].get("setup"),
        )

    checkpoints = [
        Checkpoint(url=cp["url"], path=cp["path"])
        for cp in data.get("checkpoints", [])
    ]

    convert_steps = [
        ConvertStep(script=step["script"], args=step.get("args", {}))
        for step in data.get("convert", [])
    ]

    demo_data = data.get("demo", {})
    demo = DemoConfig(image_args=demo_data.get("image_args", {}))

    skip = (model_dir / ".skip").exists()

    return ModelConfig(
        id=data["id"],
        name=data["name"],
        description=data["description"],
        task=data["task"],
        type=data.get("type", "single"),
        arch=data.get("arch", "generic"),
        tiling=data.get("tiling", False),
        dep_group=data.get("dep_group", "core"),
        skip=skip,
        repo=repo,
        checkpoints=checkpoints,
        convert=convert_steps,
        demo=demo,
        model_dir=model_dir,
        root_dir=root_dir,
    )
