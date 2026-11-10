"""YAML loaders for the film experiment."""

from __future__ import annotations

from pathlib import Path

import yaml

from sensor_sim.film import ROOT

CONFIG_DIR = ROOT / "configs" / "film"


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def scene_config(path: Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "scene.yaml")


def rendering_config(path: Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "rendering.yaml")


def dataset_config(path: Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "dataset.yaml")


def train_config(path: Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "train.yaml")


def write_yaml(path: Path, data: dict) -> None:
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
