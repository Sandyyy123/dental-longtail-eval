"""Class-distribution analysis for long-tailed detection/segmentation datasets.

Parses either a COCO-format JSON or a directory of YOLO polygon label files and
reports per-class instance counts plus the LVIS frequency grouping.

LVIS grouping convention (Gupta, Dollar & Girshick, CVPR 2019, arXiv:1904.03797):
    rare      : appears in  <  10 training images
    common    : appears in 10-100 training images
    frequent  : appears in  > 100 training images

We report the grouping on *image* counts (the LVIS definition) and also on raw
instance counts, because a dataset can have many instances concentrated in very
few images, which behaves like a rare class at training time.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

RARE_MAX = 10
COMMON_MAX = 100


@dataclass
class ClassStat:
    class_id: int
    name: str
    n_instances: int = 0
    n_images: int = 0

    @property
    def group(self) -> str:
        """LVIS frequency group, keyed on image count."""
        if self.n_images < RARE_MAX:
            return "rare"
        if self.n_images <= COMMON_MAX:
            return "common"
        return "frequent"


@dataclass
class DatasetStats:
    split: str
    n_images: int
    classes: Dict[int, ClassStat] = field(default_factory=dict)

    @property
    def n_instances(self) -> int:
        return sum(c.n_instances for c in self.classes.values())

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of the largest class to the smallest *non-empty* class."""
        counts = [c.n_instances for c in self.classes.values() if c.n_instances > 0]
        if not counts:
            return 0.0
        return max(counts) / min(counts)

    def by_group(self) -> Dict[str, List[ClassStat]]:
        out: Dict[str, List[ClassStat]] = {"rare": [], "common": [], "frequent": []}
        for c in self.classes.values():
            out[c.group].append(c)
        for g in out:
            out[g].sort(key=lambda c: c.n_instances, reverse=True)
        return out

    def empty_classes(self) -> List[ClassStat]:
        """Classes declared in the label map but with zero instances in this split."""
        return sorted(
            (c for c in self.classes.values() if c.n_instances == 0),
            key=lambda c: c.class_id,
        )


def from_coco(path: str, split: str = "unknown") -> DatasetStats:
    """Build DatasetStats from a COCO-format annotation JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)

    stats = DatasetStats(split=split, n_images=len(coco.get("images", [])))
    for cat in coco.get("categories", []):
        stats.classes[cat["id"]] = ClassStat(class_id=cat["id"], name=cat["name"])

    seen_pairs = set()
    for ann in coco.get("annotations", []):
        cid = ann["category_id"]
        if cid not in stats.classes:
            stats.classes[cid] = ClassStat(class_id=cid, name=f"class_{cid}")
        stats.classes[cid].n_instances += 1
        pair = (cid, ann["image_id"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            stats.classes[cid].n_images += 1
    return stats


def from_yolo(label_dir: str, names: List[str], split: str = "unknown") -> DatasetStats:
    """Build DatasetStats from a directory of YOLO label .txt files.

    Handles both box rows (5 fields) and polygon rows (1 + 2N fields); we only
    need the leading class index, so the row length does not matter.
    """
    stats = DatasetStats(split=split, n_images=0)
    for i, name in enumerate(names):
        stats.classes[i] = ClassStat(class_id=i, name=name)

    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"label dir not found: {label_dir}")

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue
        stats.n_images += 1
        present = set()
        with open(os.path.join(label_dir, fname), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    cid = int(float(line.split()[0]))
                except (ValueError, IndexError):
                    continue
                if cid not in stats.classes:
                    stats.classes[cid] = ClassStat(class_id=cid, name=f"class_{cid}")
                stats.classes[cid].n_instances += 1
                present.add(cid)
        for cid in present:
            stats.classes[cid].n_images += 1
    return stats


def from_counts(
    counts: Dict[str, int],
    n_images: int,
    split: str = "unknown",
    images_per_class: Optional[Dict[str, int]] = None,
) -> DatasetStats:
    """Build DatasetStats directly from published per-class instance counts.

    Used when only summary statistics are available (e.g. a dataset card or a
    job spec) rather than the annotation files themselves. When per-class image
    counts are unknown we conservatively assume one instance per image for the
    frequency grouping, which is the most favourable assumption to the dataset.
    """
    stats = DatasetStats(split=split, n_images=n_images)
    for i, (name, n) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
        img_n = images_per_class.get(name, n) if images_per_class else n
        stats.classes[i] = ClassStat(
            class_id=i, name=name, n_instances=n, n_images=min(img_n, n_images)
        )
    return stats


def format_table(stats: DatasetStats, top: int = 0) -> str:
    """Render a per-class table sorted by instance count, descending."""
    rows = sorted(stats.classes.values(), key=lambda c: c.n_instances, reverse=True)
    if top:
        rows = rows[:top]
    width = max((len(c.name) for c in rows), default=10)
    out = [f"{'class'.ljust(width)}  {'instances':>10}  {'images':>8}  group"]
    out.append("-" * (width + 32))
    for c in rows:
        out.append(
            f"{c.name.ljust(width)}  {c.n_instances:>10,}  {c.n_images:>8,}  {c.group}"
        )
    return "\n".join(out)
