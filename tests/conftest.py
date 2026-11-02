"""Shared fixtures: real config files and a tiny fake IDD-style YOLO dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

# Same names/order as the Kaggle IDD-Detection YOLO mirror (alphabetical, 15 classes).
IDD_NAMES = [
    "animal", "autorickshaw", "bicycle", "bus", "car", "caravan", "motorcycle", "person",
    "rider", "traffic light", "traffic sign", "trailer", "train", "truck", "vehicle fallback",
]


@pytest.fixture
def classes_yaml() -> Path:
    return REPO / "configs" / "classes.yaml"


@pytest.fixture
def data_yaml() -> Path:
    return REPO / "configs" / "data.yaml"


def _box(cls: int) -> str:
    return f"{cls} 0.5 0.5 0.1 0.2"


@pytest.fixture
def fake_idd(tmp_path: Path) -> Path:
    """Fake dataset: per split, images with known classes (source ids)."""
    root = tmp_path / "input" / "IDDDetectionsYOLODataset"
    layout = {
        # name: list of source class ids (None = no label file at all)
        "train": {
            "t_auto": [1, 4, 8],        # autorickshaw, car, rider
            "t_cyc": [2, 8],            # bicycle, rider
            "t_car1": [4, 4],
            "t_car2": [4, 13],          # car, truck
            "t_car3": [3, 6, 7],        # bus, motorcycle, person
            "t_car4": [5, 11],          # caravan->car, trailer->truck
            "t_sign": [10, 9],          # only dropped classes -> background
            "t_nolabel": None,          # no label file -> background
        },
        "val": {"v1": [1], "v2": [4, 7]},
        "test": {"s1": [6, 6], "s2": [0, 12, 14]},  # s2 -> background
    }
    for split, items in layout.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name, ids in items.items():
            (root / split / "images" / f"{name}.jpg").write_bytes(b"\xff\xd8fake")
            if ids is not None:
                (root / split / "labels" / f"{name}.txt").write_text("\n".join(_box(i) for i in ids) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"path": "/somewhere/else", "train": "train/images",
                                                    "val": "val/images", "test": "test/images",
                                                    "nc": len(IDD_NAMES), "names": IDD_NAMES}))
    return root
