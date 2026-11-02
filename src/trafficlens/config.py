"""Typed loading of the YAML configuration files in ``configs/``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict (an empty file gives an empty dict)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


@dataclass(frozen=True)
class SourceMapping:
    """How one source dataset's class names map onto the TrafficLens classes."""

    map: dict[str, str]
    drop: list[str] = field(default_factory=list)
    drop_unlisted: bool = False


@dataclass(frozen=True)
class ClassesConfig:
    """Final class list (index = YOLO class id) plus per-source mappings."""

    classes: list[str]
    sources: dict[str, SourceMapping]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassesConfig":
        raw = load_yaml(path)
        sources = {
            name: SourceMapping(
                map=dict(spec.get("map", {})),
                drop=list(spec.get("drop", [])),
                drop_unlisted=bool(spec.get("drop_unlisted", False)),
            )
            for name, spec in raw.get("sources", {}).items()
        }
        return cls(classes=list(raw["classes"]), sources=sources)

    def source(self, name: str) -> SourceMapping:
        """Return the mapping for source ``name`` or raise a clear error."""
        if name not in self.sources:
            raise KeyError(f"No mapping for source '{name}' in classes config; have {sorted(self.sources)}")
        return self.sources[name]


@dataclass(frozen=True)
class DataConfig:
    """Settings for Phase 1 data preparation (see ``configs/data.yaml``)."""

    source_name: str
    source_root: str | None
    search_dir: str
    images_dir: str
    labels_dir: str
    splits: dict[str, str]
    dataset_dir: str
    reports_dir: str
    link_mode: str
    seed: int
    max_images: dict[str, int | None]
    keep_all_with: list[str]
    max_background_frac: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        raw = load_yaml(path)
        src, out, sub = raw["source"], raw["output"], raw["subset"]
        link_mode = out.get("link_mode", "symlink")
        if link_mode not in {"symlink", "copy"}:
            raise ValueError(f"output.link_mode must be 'symlink' or 'copy', got {link_mode!r}")
        return cls(
            source_name=src["name"],
            source_root=src.get("root"),
            search_dir=src.get("search_dir", "/kaggle/input"),
            images_dir=src.get("images_dir", "{split}/images"),
            labels_dir=src.get("labels_dir", "{split}/labels"),
            splits=dict(src.get("splits", {"train": "train", "val": "val", "test": "test"})),
            dataset_dir=out["dataset_dir"],
            reports_dir=out["reports_dir"],
            link_mode=link_mode,
            seed=int(sub.get("seed", 42)),
            max_images=dict(sub.get("max_images", {})),
            keep_all_with=list(sub.get("keep_all_with", [])),
            max_background_frac=float(sub.get("max_background_frac", 0.05)),
        )
