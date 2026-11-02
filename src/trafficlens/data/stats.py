"""Class distribution statistics and plots for a prepared dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from trafficlens.data.prepare import ImageRecord

# Reference palette (light surface); single-hue bars, since identity is carried by axis labels.
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_2 = "#52514e"
_GRID = "#e4e3df"
_BAR = "#2a78d6"


def split_stats(splits: Mapping[str, Sequence["ImageRecord"]], class_names: Sequence[str]) -> pd.DataFrame:
    """Per split and class: instance count, number of images containing it, and share.

    Returns:
        Long-format DataFrame with columns
        ``split, class_id, class, instances, images, instance_share``.
    """
    rows = []
    for split, records in splits.items():
        inst = [0] * len(class_names)
        imgs = [0] * len(class_names)
        for r in records:
            for cid, n in r.class_counts.items():
                inst[cid] += n
                imgs[cid] += 1
        total = sum(inst) or 1
        for cid, name in enumerate(class_names):
            rows.append(
                {
                    "split": split,
                    "class_id": cid,
                    "class": name,
                    "instances": inst[cid],
                    "images": imgs[cid],
                    "instance_share": inst[cid] / total,
                }
            )
    return pd.DataFrame(rows)


def imbalance_ratio(stats: pd.DataFrame, split: str) -> float:
    """Largest / smallest non-zero instance count in ``split`` (inf if a class is absent)."""
    counts = stats.loc[stats["split"] == split, "instances"]
    if (counts == 0).any():
        return float("inf")
    return float(counts.max() / counts.min())


def sample_filenames(records: Sequence["ImageRecord"], n: int = 8) -> list[str]:
    """Evenly spaced image filenames, used to check whether names encode drive sequences."""
    if not records:
        return []
    step = max(1, len(records) // n)
    return [records[i].image.name for i in range(0, len(records), step)][:n]


def plot_class_distribution(stats: pd.DataFrame, path: str | Path) -> Path:
    """Save one horizontal bar panel per split: instances per class, sorted by train count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(dict.fromkeys(stats["split"]))
    first = stats[stats["split"] == splits[0]]
    order = list(first.sort_values("instances")["class"])

    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.8), sharey=True, facecolor=_SURFACE)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        sub = stats[stats["split"] == split].set_index("class").loc[order]
        ax.set_facecolor(_SURFACE)
        ax.barh(order, sub["instances"], color=_BAR, height=0.6)
        vmax = max(int(sub["instances"].max()), 1)
        for y, (v, share) in enumerate(zip(sub["instances"], sub["instance_share"])):
            ax.text(v + vmax * 0.02, y, f"{v:,}  ({share:.1%})", va="center", fontsize=8, color=_TEXT_2)
        ax.set_xlim(0, vmax * 1.45)
        ax.set_title(f"{split}", loc="left", fontsize=11, color=_TEXT, fontweight="bold")
        ax.tick_params(colors=_TEXT_2, labelsize=9, length=0)
        ax.grid(axis="x", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.set_xlabel("instances", color=_TEXT_2, fontsize=9)
    fig.suptitle("TrafficLens dataset: instances per class", x=0.01, ha="left", fontsize=12, color=_TEXT)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=_SURFACE)
    plt.close(fig)
    return path
