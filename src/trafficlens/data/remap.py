"""Class remapping from a source dataset's classes to the TrafficLens class list.

Everything here is pure (no file I/O) so it can be unit-tested directly. The same
remap is used for dataset labels (Phase 1) and for COCO baseline predictions (Phase 2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trafficlens.config import SourceMapping


class UnknownClassError(ValueError):
    """A source class is neither mapped nor explicitly dropped."""


def normalize_name(name: str) -> str:
    """Normalise a class name for comparison: lower-case, ``_``/``-`` → space, single spaces."""
    return " ".join(name.replace("_", " ").replace("-", " ").lower().split())


def names_to_list(names: Sequence[str] | Mapping[int, str]) -> list[str]:
    """Convert YOLO ``names`` (list, or dict ``{id: name}``) to a list indexed by class id."""
    if isinstance(names, Mapping):
        by_id = {int(k): str(v) for k, v in names.items()}
        ids = sorted(by_id)
        if ids != list(range(len(ids))):
            raise ValueError(f"Class ids must be contiguous from 0, got {ids}")
        return [by_id[i] for i in ids]
    return [str(n) for n in names]


@dataclass(frozen=True)
class ClassRemap:
    """Result of matching a source class list against the target class list.

    Attributes:
        source_names: Source class names, index = source id.
        target_names: Target class names, index = target id.
        id_map: Source id → target id, for kept classes only.
        dropped: Source class names that are discarded.
    """

    source_names: list[str]
    target_names: list[str]
    id_map: dict[int, int]
    dropped: list[str]

    def lookup_table(self) -> np.ndarray:
        """Array ``t`` with ``t[source_id] = target_id``, or -1 for dropped classes."""
        table = np.full(len(self.source_names), -1, dtype=np.int64)
        for s, t in self.id_map.items():
            table[s] = t
        return table


def build_remap(
    source_names: Sequence[str] | Mapping[int, str],
    target_names: Sequence[str],
    mapping: SourceMapping,
) -> ClassRemap:
    """Match source class names to target classes using ``mapping``.

    Args:
        source_names: The source dataset's class names (list or ``{id: name}``).
        target_names: The final TrafficLens classes, in id order.
        mapping: Name-based mapping and drop list for this source.

    Returns:
        A :class:`ClassRemap`.

    Raises:
        UnknownClassError: A source class is neither mapped nor dropped
            (unless ``mapping.drop_unlisted`` is set).
        ValueError: The mapping points at a class that isn't in ``target_names``.
    """
    src = names_to_list(source_names)
    target_index = {normalize_name(n): i for i, n in enumerate(target_names)}
    name_map = {normalize_name(k): normalize_name(v) for k, v in mapping.map.items()}
    drop = {normalize_name(d) for d in mapping.drop}

    for tgt in name_map.values():
        if tgt not in target_index:
            raise ValueError(f"Mapping targets unknown class '{tgt}'; target classes are {list(target_names)}")

    id_map: dict[int, int] = {}
    dropped: list[str] = []
    unknown: list[str] = []
    for sid, raw in enumerate(src):
        name = normalize_name(raw)
        if name in name_map:
            id_map[sid] = target_index[name_map[name]]
        elif name in drop or mapping.drop_unlisted:
            dropped.append(raw)
        else:
            unknown.append(raw)
    if unknown:
        raise UnknownClassError(
            f"Source classes {unknown} are neither mapped nor dropped. "
            "Add them to configs/classes.yaml (map or drop) after checking what they are."
        )
    return ClassRemap(source_names=src, target_names=list(target_names), id_map=id_map, dropped=dropped)


def remap_label_lines(lines: Iterable[str], id_map: Mapping[int, int]) -> list[str]:
    """Remap the class ids in YOLO label lines (``cls x y w h [...]``).

    Lines whose class is not in ``id_map`` are dropped; blank lines are skipped.
    Coordinates are passed through unchanged (as text, so there is no rounding).

    Raises:
        ValueError: A line is malformed (fewer than 5 fields or a non-integer class id).
    """
    out: list[str] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Malformed YOLO label line (need >= 5 fields): {line!r}")
        try:
            cls_id = int(float(parts[0]))
        except ValueError as e:
            raise ValueError(f"Non-numeric class id in label line: {line!r}") from e
        if cls_id in id_map:
            out.append(" ".join([str(id_map[cls_id]), *parts[1:]]))
    return out


def remap_class_ids(class_ids: np.ndarray, remap: ClassRemap) -> tuple[np.ndarray, np.ndarray]:
    """Remap an array of source class ids (e.g. COCO predictions).

    Returns:
        ``(new_ids, keep)`` where ``keep`` is a boolean mask of entries with a target
        class and ``new_ids`` holds the target ids for those entries only.
    """
    class_ids = np.asarray(class_ids, dtype=np.int64)
    table = remap.lookup_table()
    valid = (class_ids >= 0) & (class_ids < len(table))
    mapped = np.full(class_ids.shape, -1, dtype=np.int64)
    mapped[valid] = table[class_ids[valid]]
    keep = mapped >= 0
    return mapped[keep], keep
