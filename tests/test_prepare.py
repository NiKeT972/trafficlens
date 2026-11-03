"""End-to-end data preparation on a tiny fake IDD-style dataset."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from trafficlens.config import ClassesConfig, DataConfig
from trafficlens.data.prepare import ImageRecord, find_source_root, prepare, select_subset


def _cfg(data_yaml: Path, root: Path, tmp_path: Path, **kw) -> DataConfig:
    cfg = DataConfig.from_yaml(data_yaml)
    return replace(cfg, source_root=None, search_dir=str(root.parent), dataset_dir=str(tmp_path / "out"),
                   reports_dir=str(tmp_path / "reports"), **kw)


def test_find_source_root(fake_idd):
    assert find_source_root(fake_idd.parent.parent) == fake_idd


def test_find_source_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_source_root(tmp_path)


def test_prepare_end_to_end(fake_idd, data_yaml, classes_yaml, tmp_path):
    cfg = _cfg(data_yaml, fake_idd, tmp_path, max_images={"train": None, "val": None, "test": None},
               max_background_frac=0.5)
    summary = prepare(cfg, ClassesConfig.from_yaml(classes_yaml))
    out = tmp_path / "out"

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "car", 1: "bus", 2: "truck", 3: "auto_rickshaw", 4: "two_wheeler",
                             5: "bicycle", 6: "pedestrian"}
    assert data["train"] == "train/images" and Path(data["path"]).is_absolute()

    # Images are symlinks to the source; labels are remapped.
    link = out / "train" / "images" / "t_auto.jpg"
    assert link.is_symlink() and link.resolve() == (fake_idd / "train" / "images" / "t_auto.jpg").resolve()
    assert (out / "train" / "labels" / "t_auto.txt").read_text().split("\n")[:2] == [
        "3 0.5 0.5 0.1 0.2", "0 0.5 0.5 0.1 0.2"]  # autorickshaw->3, car->0, rider dropped
    assert (out / "train" / "labels" / "t_car4.txt").read_text().split() [0::5] == ["0", "2"]  # caravan, trailer

    assert summary["images_source"] == {"train": 8, "val": 2, "test": 2}
    assert summary["missing_label_files"]["train"] == 1
    for name in ("class_stats_source.csv", "class_stats_prepared.csv", "class_distribution.png", "summary.json"):
        assert (tmp_path / "reports" / name).is_file()
    assert json.loads((tmp_path / "reports" / "summary.json").read_text())["dropped"]


def _rec(name: str, ids: list[int]) -> ImageRecord:
    return ImageRecord(image=Path(f"/x/{name}.jpg"), lines=[f"{i} 0 0 0 0" for i in ids], class_counts=Counter(ids))


def test_select_subset_keeps_rare_and_caps_background():
    recs = [_rec(f"car{i}", [0]) for i in range(50)] + [_rec(f"auto{i}", [3]) for i in range(5)] + \
           [_rec(f"bg{i}", []) for i in range(20)]
    sel = select_subset(recs, max_images=20, keep_all_ids={3}, max_background_frac=0.1, seed=0)
    names = [r.image.stem for r in sel]
    assert len(sel) == 20
    assert all(f"auto{i}" in names for i in range(5))
    assert sum(n.startswith("bg") for n in names) == 2
    # deterministic
    assert [r.image for r in select_subset(recs, 20, {3}, 0.1, 0)] == [r.image for r in sel]


def test_select_subset_no_cap_limits_background_fraction():
    recs = [_rec(f"c{i}", [0]) for i in range(19)] + [_rec(f"bg{i}", []) for i in range(10)]
    sel = select_subset(recs, max_images=None, keep_all_ids=set(), max_background_frac=0.05, seed=0)
    assert sum(r.is_background for r in sel) == 1  # 1 / 20 = 5%


def test_find_source_root_kaggle_nested_layout(tmp_path):
    # New Kaggle layout: /kaggle/input/datasets/<owner>/<slug>/<folder>/data.yaml
    root = tmp_path / "input" / "datasets" / "redzapdos123" / "indian-driving-dataset-detections-yolov11" / "IDDDetectionsYOLODataset"
    (root / "train" / "images").mkdir(parents=True)
    (root / "data.yaml").write_text("names: [car]\n")
    assert find_source_root(tmp_path / "input") == root
