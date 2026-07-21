"""Boundary IoU (Cheng, Girshick, Dollar, Schwing & Kirillov, CVPR 2021).

Reference: "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation"
arXiv:2103.16562

Why this matters for dental radiographs specifically. Standard mask IoU is
dominated by interior pixels, so it becomes progressively less sensitive to
boundary error as objects get larger. The large anatomical classes here
(Mandibular Canal, maxillary sinus, Bone Loss) are exactly the ones where a
clinician judges the mask by its border, and exactly the ones where mask IoU is
least able to see a bad border. A model can gain mask mAP while its canal
tracings get visibly worse.

Boundary IoU restricts the comparison to a thin band along each mask's contour,
so it scores what the eye is actually checking. It is the metric used by the
LVIS challenge for this reason.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import binary_erosion

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    _HAS_SCIPY = False


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary erosion with a 3x3 cross structuring element."""
    if iterations <= 0:
        return mask.copy()
    if _HAS_SCIPY:
        return binary_erosion(mask, iterations=iterations, border_value=0)
    # Dependency-free fallback: iterated 4-neighbour min filter, zero-padded so
    # the image border erodes inward exactly as scipy's border_value=0 does.
    out = mask.copy()
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        )
    return out


def boundary_region(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """The set of pixels within `d` of the mask contour, intersected with the mask.

    `d` is `dilation_ratio` times the image diagonal, matching the paper.
    """
    mask = mask.astype(bool)
    h, w = mask.shape
    diag = float(np.sqrt(h * h + w * w))
    d = int(round(dilation_ratio * diag))
    d = max(d, 1)
    return mask & ~_erode(mask, d)


def boundary_iou(
    gt: np.ndarray, pred: np.ndarray, dilation_ratio: float = 0.02
) -> float:
    """Boundary IoU between two binary masks of identical shape."""
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    gt_b = boundary_region(gt, dilation_ratio)
    pred_b = boundary_region(pred, dilation_ratio)

    inter = np.count_nonzero(gt_b & pred_b)
    union = np.count_nonzero(gt_b | pred_b)
    return float(inter) / float(union) if union else 0.0


def mask_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Standard mask IoU, for side-by-side comparison."""
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    union = np.count_nonzero(gt | pred)
    return float(np.count_nonzero(gt & pred)) / float(union) if union else 0.0
