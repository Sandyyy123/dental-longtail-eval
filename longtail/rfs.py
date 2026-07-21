"""Repeat Factor Sampling (Gupta, Dollar & Girshick, CVPR 2019, arXiv:1904.03797).

The oversampling scheme introduced with LVIS and the base that every stronger
long-tail method in the literature (EQLv2, Seesaw, Federated Loss, ECM Loss)
is reported on top of. It is also the cheapest thing to try first: no loss
surgery, no architecture change, purely a change to how the sampler draws.

Per class c with image-frequency f(c) (fraction of training images containing
at least one instance of c):

    r(c) = max(1, sqrt(t / f(c)))

Per image i, the repeat factor is the max over the classes it contains:

    r(i) = max over c in i of r(c)

Images are then repeated r(i) times per epoch, with the fractional part drawn
stochastically each epoch so the expectation is exact.

Note the design choice that matters: taking the max over classes means an image
containing both a head class and a tail class is oversampled at the tail rate.
That is intended, but it does mean head-class instances get oversampled as a
side effect, which is one reason RFS alone plateaus and the loss-based methods
add value on top.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


def class_image_frequencies(
    image_classes: Sequence[Iterable[int]],
) -> Dict[int, float]:
    """f(c): fraction of images containing at least one instance of class c."""
    n_images = len(image_classes)
    if n_images == 0:
        return {}
    counts: Dict[int, int] = defaultdict(int)
    for classes in image_classes:
        for c in set(classes):
            counts[c] += 1
    return {c: n / n_images for c, n in counts.items()}


def class_repeat_factors(
    freqs: Dict[int, float], threshold: float = 0.001
) -> Dict[int, float]:
    """r(c) = max(1, sqrt(t / f(c)))."""
    return {
        c: max(1.0, math.sqrt(threshold / f)) if f > 0 else 1.0
        for c, f in freqs.items()
    }


def image_repeat_factors(
    image_classes: Sequence[Iterable[int]], threshold: float = 0.001
) -> List[float]:
    """r(i) = max over classes present in image i."""
    freqs = class_image_frequencies(image_classes)
    r_c = class_repeat_factors(freqs, threshold)
    out = []
    for classes in image_classes:
        cs = set(classes)
        out.append(max((r_c.get(c, 1.0) for c in cs), default=1.0))
    return out


def expected_epoch_size(repeat_factors: Sequence[float]) -> float:
    """Expected number of samples drawn per epoch under RFS."""
    return float(sum(repeat_factors))


def sample_epoch_indices(
    repeat_factors: Sequence[float], rng
) -> List[int]:
    """Materialize one epoch's index list.

    The integer part of r(i) is deterministic; the fractional part is a Bernoulli
    draw, so E[repeats] == r(i) exactly.
    """
    indices: List[int] = []
    for i, r in enumerate(repeat_factors):
        n = int(math.floor(r))
        if rng.random() < (r - n):
            n += 1
        indices.extend([i] * n)
    return indices
