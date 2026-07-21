"""Verified facts about the 31-class dental panoramic dataset.

Every number here is traceable to a stated source. Where a value was not
published, it is recorded as None rather than estimated, because the whole point
of this toolkit is that unstated tail counts are where the evaluation breaks.

Sources
-------
SPEC   : the job specification as written by the client (split sizes, the four
         per-class counts it quotes verbatim).
ROBOFLOW: the dataset's Roboflow Universe listing,
         universe.roboflow.com/celldetection-ok5sm/dental-x-ray-panoramic-dataset
         CC BY 4.0, 13,814 images, published Feb 2025. Provides the class names.

Known discrepancy, recorded rather than smoothed over: the split sizes in SPEC
sum to 13,932 images, while the Roboflow project lists 13,814. The client is
working from a re-exported version of the project, so the two are not expected
to match exactly. Any analysis below that depends on split size is therefore
labelled as being on SPEC's numbers.
"""

from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------- split sizes
# Source: SPEC
SPLITS = {
    "train": {"images": 9_481, "annotations": 94_794},
    "valid": {"images": 2_871, "annotations": 27_141},
    "test": {"images": 1_580, "annotations": 14_957},
}

TOTAL_IMAGES = sum(s["images"] for s in SPLITS.values())          # 13,932
TOTAL_ANNOTATIONS = sum(s["annotations"] for s in SPLITS.values())  # 136,892
TEST_FRACTION = SPLITS["test"]["annotations"] / TOTAL_ANNOTATIONS

# --------------------------------------------------------------- class names
# Source: ROBOFLOW. All 31 names, verbatim, including the inconsistent casing
# and the spaced hyphen in "post - core", which are properties of the dataset.
CLASS_NAMES = [
    "wire", "abutment", "attrition", "bone defect", "Bone Loss", "Caries",
    "Crown", "Cyst", "Filling", "Fracture teeth", "gingival former",
    "impacted tooth", "Implant", "Malaligned", "Mandibular Canal",
    "maxillary sinus", "metal band", "Missing teeth", "orthodontic brackets",
    "Periapical lesion", "permanent retainer", "Permanent Teeth", "plating",
    "post - core", "Primary teeth", "Retained root", "Root Canal Treatment",
    "Root Piece", "Root resorption", "Supra Eruption", "TAD",
]

# ------------------------------------------------------- published per-class
# Source: SPEC, quoted verbatim. Only these four classes have a stated count.
# NOTE: SPEC does not say whether these are train-split or whole-dataset totals.
# Both readings are handled in main.py; both lead to the same conclusion, and
# the train-only reading is the *worse* case for test-set coverage.
PUBLISHED_COUNTS: Dict[str, Optional[int]] = {
    "Fracture teeth": 9,
    "TAD": 4,
    "bone defect": 1,
}

# SPEC states these three sit in the 8,000-33,000 range but does not give exact
# values, so the range is recorded and the point estimate is not invented.
PUBLISHED_HEAD_RANGE = (8_000, 33_000)
HEAD_CLASSES = ["Filling", "impacted tooth", "Root Canal Treatment"]

# Every other class: count not published.
UNKNOWN_COUNT_CLASSES = [
    c for c in CLASS_NAMES
    if c not in PUBLISHED_COUNTS and c not in HEAD_CLASSES
]


def stated_imbalance_ratio() -> float:
    """Worst-case ratio the client's own description implies: 33,000 : 1."""
    return PUBLISHED_HEAD_RANGE[1] / min(
        v for v in PUBLISHED_COUNTS.values() if v is not None
    )


def coverage_note() -> str:
    return (
        f"{len(PUBLISHED_COUNTS) + len(HEAD_CLASSES)} of {len(CLASS_NAMES)} class "
        f"counts are published (3 exactly, 3 as a range); "
        f"{len(UNKNOWN_COUNT_CLASSES)} are unstated."
    )
