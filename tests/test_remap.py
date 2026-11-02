"""Class remapping: IDD and COCO names onto the TrafficLens classes."""

from __future__ import annotations

import numpy as np
import pytest

from trafficlens.config import ClassesConfig, SourceMapping
from trafficlens.data.remap import (
    UnknownClassError,
    build_remap,
    names_to_list,
    normalize_name,
    remap_class_ids,
    remap_label_lines,
)

from conftest import IDD_NAMES

TARGET = ["car", "bus", "truck", "auto_rickshaw", "two_wheeler", "bicycle", "pedestrian"]


def test_normalize_name():
    assert normalize_name("Auto-Rickshaw") == "auto rickshaw"
    assert normalize_name("  traffic_light ") == "traffic light"
    assert normalize_name("vehicle   fallback") == "vehicle fallback"


def test_names_to_list_accepts_dict_with_int_or_str_keys():
    assert names_to_list({1: "b", 0: "a"}) == ["a", "b"]
    assert names_to_list({"0": "a", "1": "b"}) == ["a", "b"]
    with pytest.raises(ValueError):
        names_to_list({0: "a", 2: "c"})


def test_idd_remap_from_real_config(classes_yaml):
    cfg = ClassesConfig.from_yaml(classes_yaml)
    assert cfg.classes == TARGET
    r = build_remap(IDD_NAMES, cfg.classes, cfg.source("idd"))
    by_name = {r.source_names[s]: cfg.classes[t] for s, t in r.id_map.items()}
    assert by_name == {
        "autorickshaw": "auto_rickshaw",
        "bicycle": "bicycle",
        "bus": "bus",
        "car": "car",
        "caravan": "car",
        "motorcycle": "two_wheeler",
        "person": "pedestrian",
        "trailer": "truck",
        "truck": "truck",
    }
    assert set(r.dropped) == {"animal", "rider", "traffic light", "traffic sign", "train", "vehicle fallback"}


def test_unknown_source_class_fails_loudly():
    mapping = SourceMapping(map={"car": "car"}, drop=["sign"])
    with pytest.raises(UnknownClassError, match="scooter"):
        build_remap(["car", "sign", "scooter"], TARGET, mapping)


def test_mapping_to_unknown_target_fails():
    with pytest.raises(ValueError, match="unknown class"):
        build_remap(["car"], TARGET, SourceMapping(map={"car": "sedan"}))


def test_drop_unlisted_for_coco(classes_yaml):
    cfg = ClassesConfig.from_yaml(classes_yaml)
    coco = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    r = build_remap(coco, cfg.classes, cfg.source("coco"))
    assert r.id_map == {0: 6, 1: 5, 2: 0, 3: 4, 5: 1, 7: 2}
    assert "auto_rickshaw" not in {cfg.classes[t] for t in r.id_map.values()}


def test_remap_label_lines_keeps_coords_and_drops_unmapped():
    lines = ["4 0.1 0.2 0.3 0.4", "", "8 0.5 0.5 0.1 0.1", "1 0.123456789 0.2 0.3 0.4"]
    out = remap_label_lines(lines, {4: 0, 1: 3})
    assert out == ["0 0.1 0.2 0.3 0.4", "3 0.123456789 0.2 0.3 0.4"]


def test_remap_label_lines_rejects_malformed():
    with pytest.raises(ValueError, match="Malformed"):
        remap_label_lines(["4 0.1 0.2"], {4: 0})
    with pytest.raises(ValueError, match="Non-numeric"):
        remap_label_lines(["car 0.1 0.2 0.3 0.4"], {4: 0})


def test_remap_class_ids_array():
    r = build_remap(["a", "b", "c"], ["x", "y"], SourceMapping(map={"a": "y", "c": "x"}, drop=["b"]))
    new, keep = remap_class_ids(np.array([0, 1, 2, 2, 7]), r)
    assert keep.tolist() == [True, False, True, True, False]
    assert new.tolist() == [1, 0, 0]
