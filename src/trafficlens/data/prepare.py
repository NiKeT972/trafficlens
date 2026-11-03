"""Phase 1 data preparation: remap classes, subsample and write a YOLO dataset.

Source: a YOLO-format dataset (default: the Kaggle mirror of IDD-Detection), laid out as
``<root>/<split>/images`` and ``<root>/<split>/labels`` with a ``data.yaml`` in ``<root>``.

Output (``output.dataset_dir``)::

    <dataset_dir>/
        data.yaml                 # TrafficLens classes, paths to the splits
        train/images/*.jpg        # symlinks to the source images (no copies)
        train/labels/*.txt        # labels with remapped class ids
        val/...  test/...

Images are symlinked by default so the prepared dataset fits in Kaggle's 20 GB
``/kaggle/working`` even though the source is about 22 GB.

Usage::

    python -m trafficlens.data.prepare --data-config configs/data.yaml \
        --classes-config configs/classes.yaml [--source-root PATH] [--dataset-dir PATH] [--reports-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml
from tqdm import tqdm

from trafficlens.config import ClassesConfig, DataConfig, load_yaml
from trafficlens.data.remap import ClassRemap, build_remap, remap_label_lines

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    """One image and its remapped labels.

    Attributes:
        image: Path to the source image.
        lines: Remapped YOLO label lines (empty = background image).
        class_counts: Target class id → number of instances in this image.
        had_label_file: Whether the source had a label file for this image.
    """

    image: Path
    lines: list[str]
    class_counts: Counter[int] = field(default_factory=Counter)
    had_label_file: bool = True

    @property
    def is_background(self) -> bool:
        return not self.lines


def find_source_root(search_dir: str | Path, max_depth: int = 6) -> Path:
    """Find the folder holding the source ``data.yaml`` under ``search_dir``.

    Kaggle mounts datasets at ``/kaggle/input/datasets/<owner>/<slug>/...``, so the
    default depth is generous. ``images``/``labels`` folders are never entered (they are
    huge), so this stays fast. Only searches ``max_depth`` levels deep (a full
    recursive glob would be slow).

    Raises:
        FileNotFoundError: No ``data.yaml`` found.
        RuntimeError: More than one found (set ``source.root`` explicitly).
    """
    base = Path(search_dir)
    found: list[Path] = []
    frontier = [base]
    for _ in range(max_depth + 1):
        nxt: list[Path] = []
        for d in frontier:
            if (d / "data.yaml").is_file():
                found.append(d)
                continue
            try:
                nxt.extend(p for p in d.iterdir() if p.is_dir() and p.name not in {"images", "labels"})
            except (PermissionError, FileNotFoundError):
                continue
        frontier = nxt
    if not found:
        raise FileNotFoundError(f"No data.yaml found within {max_depth} levels of {base}. Is the dataset attached?")
    if len(found) > 1:
        raise RuntimeError(f"Several data.yaml files found: {found}. Set source.root in configs/data.yaml.")
    return found[0]


def read_source_names(root: Path) -> list[str] | dict[int, str]:
    """Read the class names from the source ``data.yaml``."""
    raw = load_yaml(root / "data.yaml")
    if "names" not in raw:
        raise KeyError(f"{root / 'data.yaml'} has no 'names' entry")
    return raw["names"]


def _read_record(img: Path, labels_dir: Path, existing_labels: set[str], remap: ClassRemap) -> ImageRecord:
    """Build the :class:`ImageRecord` for one image (reads its label file if present)."""
    name = f"{img.stem}.txt"
    if name in existing_labels:
        text = (labels_dir / name).read_text(encoding="utf-8")
        lines = remap_label_lines(text.splitlines(), remap.id_map)
        had = True
    else:
        lines, had = [], False
    counts: Counter[int] = Counter(int(line.split(maxsplit=1)[0]) for line in lines)
    return ImageRecord(image=img, lines=lines, class_counts=counts, had_label_file=had)


def scan_split(images_dir: Path, labels_dir: Path, remap: ClassRemap, workers: int = 32) -> list[ImageRecord]:
    """Read every image in ``images_dir`` and remap its label file.

    Images without a label file are treated as background (YOLO convention).
    Label files are read with a thread pool: Kaggle's dataset mount fetches files over
    the network with high per-file latency, so sequential reads manage only a few files
    per second while parallel reads are many times faster.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    # One directory listing instead of one existence check per image.
    existing = {p.name for p in labels_dir.iterdir()} if labels_dir.is_dir() else set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = pool.map(lambda img: _read_record(img, labels_dir, existing, remap), images, chunksize=64)
        return list(tqdm(it, total=len(images), desc=f"scan {images_dir.parent.name}", unit="img", mininterval=2))


def select_subset(
    records: list[ImageRecord],
    max_images: int | None,
    keep_all_ids: set[int],
    max_background_frac: float,
    seed: int,
) -> list[ImageRecord]:
    """Pick a deterministic subset of a split.

    Rules, in order:

    1. Background images are capped at ``max_background_frac`` of the result.
    2. If ``max_images`` is None, everything else is kept.
    3. Otherwise every image containing a class in ``keep_all_ids`` is kept (even if
       that alone exceeds ``max_images``; a warning is logged), and the remaining
       budget is filled with a random sample of the other labelled images.

    Returns:
        Selected records, sorted by image path.
    """
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda r: str(r.image))
    background = [r for r in ordered if r.is_background]
    labelled = [r for r in ordered if not r.is_background]

    if max_images is None:
        chosen = labelled
        # bg / (labelled + bg) <= frac  <=>  bg <= frac / (1 - frac) * labelled
        n_bg = min(len(background), int(len(labelled) * max_background_frac / max(1e-9, 1 - max_background_frac)))
    else:
        n_bg = min(len(background), int(max_images * max_background_frac))
        priority = [r for r in labelled if keep_all_ids & set(r.class_counts)]
        others = [r for r in labelled if not keep_all_ids & set(r.class_counts)]
        if len(priority) > max_images - n_bg:
            log.warning(
                "%d images contain must-keep classes, more than the max_images=%d budget; keeping all of them",
                len(priority),
                max_images,
            )
        n_fill = max(0, max_images - n_bg - len(priority))
        chosen = priority + rng.sample(others, min(n_fill, len(others)))

    chosen = chosen + rng.sample(background, n_bg)
    return sorted(chosen, key=lambda r: str(r.image))


def write_split(records: list[ImageRecord], out_split_dir: Path, link_mode: str = "symlink") -> None:
    """Write ``images/`` (symlinks or copies) and remapped ``labels/`` for one split."""
    img_dir, lbl_dir = out_split_dir / "images", out_split_dir / "labels"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for r in tqdm(records, desc=f"write {out_split_dir.name}", unit="img", mininterval=2):
        dst = img_dir / r.image.name
        if link_mode == "symlink":
            dst.symlink_to(os.path.abspath(r.image))  # abspath: no per-file stat on the network mount
        else:
            shutil.copy2(r.image, dst)
        (lbl_dir / f"{r.image.stem}.txt").write_text("\n".join(r.lines) + ("\n" if r.lines else ""), encoding="utf-8")


def write_dataset_yaml(dataset_dir: Path, class_names: list[str], splits: list[str]) -> Path:
    """Write the Ultralytics ``data.yaml`` for the prepared dataset."""
    data = {"path": str(dataset_dir.resolve())}
    for s in splits:
        data[s] = f"{s}/images"
    data["names"] = dict(enumerate(class_names))
    path = dataset_dir / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def prepare(data_cfg: DataConfig, classes_cfg: ClassesConfig) -> dict:
    """Run the full preparation and write stats/plots. Returns a summary dict."""
    from trafficlens.data.stats import split_stats, plot_class_distribution, sample_filenames

    root = Path(data_cfg.source_root) if data_cfg.source_root else find_source_root(data_cfg.search_dir)
    log.info("Source dataset root: %s", root)
    source_names = read_source_names(root)
    remap = build_remap(source_names, classes_cfg.classes, classes_cfg.source(data_cfg.source_name))
    log.info("Kept source classes: %s", {remap.source_names[s]: classes_cfg.classes[t] for s, t in remap.id_map.items()})
    log.info("Dropped source classes: %s", remap.dropped)

    keep_ids = {classes_cfg.classes.index(c) for c in data_cfg.keep_all_with}
    dataset_dir = Path(data_cfg.dataset_dir)
    reports_dir = Path(data_cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    full: dict[str, list[ImageRecord]] = {}
    selected: dict[str, list[ImageRecord]] = {}
    for split, src_folder in data_cfg.splits.items():
        images_dir = root / data_cfg.images_dir.format(split=src_folder)
        labels_dir = root / data_cfg.labels_dir.format(split=src_folder)
        full[split] = scan_split(images_dir, labels_dir, remap)
        missing = sum(not r.had_label_file for r in full[split])
        if missing:
            log.warning("%s: %d images have no label file (treated as background)", split, missing)
        selected[split] = select_subset(
            full[split],
            data_cfg.max_images.get(split),
            keep_ids,
            data_cfg.max_background_frac,
            data_cfg.seed,
        )
        write_split(selected[split], dataset_dir / split, data_cfg.link_mode)

    yaml_path = write_dataset_yaml(dataset_dir, classes_cfg.classes, list(data_cfg.splits))

    stats_full = split_stats(full, classes_cfg.classes)
    stats_sel = split_stats(selected, classes_cfg.classes)
    stats_full.to_csv(reports_dir / "class_stats_source.csv", index=False)
    stats_sel.to_csv(reports_dir / "class_stats_prepared.csv", index=False)
    plot_class_distribution(stats_sel, reports_dir / "class_distribution.png")

    summary = {
        "source_root": str(root),
        "source_classes": remap.source_names,
        "mapping": {remap.source_names[s]: classes_cfg.classes[t] for s, t in remap.id_map.items()},
        "dropped": remap.dropped,
        "images_source": {s: len(v) for s, v in full.items()},
        "images_prepared": {s: len(v) for s, v in selected.items()},
        "background_prepared": {s: sum(r.is_background for r in v) for s, v in selected.items()},
        "missing_label_files": {s: sum(not r.had_label_file for r in v) for s, v in full.items()},
        "sample_filenames": {s: sample_filenames(v) for s, v in full.items()},
        "dataset_yaml": str(yaml_path),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--classes-config", default="configs/classes.yaml")
    p.add_argument("--source-root", help="Override source.root (folder with the source data.yaml)")
    p.add_argument("--dataset-dir", help="Override output.dataset_dir")
    p.add_argument("--reports-dir", help="Override output.reports_dir")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_cfg = DataConfig.from_yaml(args.data_config)
    overrides = {
        k: v
        for k, v in {
            "source_root": args.source_root,
            "dataset_dir": args.dataset_dir,
            "reports_dir": args.reports_dir,
        }.items()
        if v
    }
    data_cfg = replace(data_cfg, **overrides)
    summary = prepare(data_cfg, ClassesConfig.from_yaml(args.classes_config))
    print(json.dumps({k: v for k, v in summary.items() if k != "sample_filenames"}, indent=2))


if __name__ == "__main__":
    main()
