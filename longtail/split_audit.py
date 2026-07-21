"""Test-split audit: which classes can a mAP number actually say anything about?

This module exists because of a failure mode that is easy to miss and fatal to
any "we beat SOTA" claim on a severely long-tailed dataset.

Two distinct problems:

1. UNDEFINED AP. `COCOeval` returns -1 for any category with no ground-truth
   instances in the evaluation set, and `-1` entries are *excluded* from the
   mean. So a headline "31-class mAP" computed on a split where 6 classes have
   no test instances is silently a 25-class mAP. The number is not wrong, it is
   answering a different question than the one it appears to answer.

2. QUANTIZED AP. With k ground-truth instances in the test split, recall can
   only take the k+1 values {0, 1/k, ..., 1}. Average precision is therefore
   quantized in steps of roughly 1/k. At k=1 the class AP is close to a coin
   flip: one detection moves it between 0 and 1. Reporting a delta of "+2.3 mAP"
   that is driven by such a class is noise, not improvement.

The audit quantifies both, so that improvement claims can be restricted to the
classes where the metric is meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

# A class needs at least this many test instances before a per-class AP is worth
# reporting. Chosen so AP quantization is <= 0.10; see quantization_step().
MIN_MEANINGFUL_INSTANCES = 10


@dataclass
class ClassAudit:
    name: str
    total_instances: int
    test_instances: Optional[int] = None
    expected_test: float = 0.0
    p_zero_in_test: float = 0.0

    @property
    def observed_or_expected(self) -> float:
        return (
            float(self.test_instances)
            if self.test_instances is not None
            else self.expected_test
        )

    @property
    def quantization_step(self) -> float:
        """Coarsest resolution of AP for this class: 1/k, or 1.0 if k == 0."""
        k = self.observed_or_expected
        return 1.0 if k < 1 else 1.0 / k

    @property
    def verdict(self) -> str:
        k = self.observed_or_expected
        if k < 1:
            return "UNDEFINED"      # COCOeval returns -1, dropped from the mean
        if k < MIN_MEANINGFUL_INSTANCES:
            return "UNRELIABLE"     # AP quantized coarser than 0.10
        return "OK"


def p_zero_in_split(total_instances: int, split_fraction: float) -> float:
    """Probability a class lands zero instances in a split, under random assignment.

    Treats each instance as independently assigned to the split with probability
    `split_fraction`. This is optimistic: real instances cluster within images,
    which makes an all-or-nothing outcome *more* likely, not less. So this is a
    lower bound on the risk.
    """
    if total_instances <= 0:
        return 1.0
    return (1.0 - split_fraction) ** total_instances


def audit(
    total_counts: Dict[str, int],
    test_fraction: float,
    observed_test_counts: Optional[Dict[str, int]] = None,
) -> List[ClassAudit]:
    """Audit every class for AP validity on the test split.

    Args:
        total_counts: per-class instance counts across the whole dataset.
        test_fraction: fraction of the dataset held out as test.
        observed_test_counts: real per-class test counts, when the split is
            available. When given, these override the expectation model.
    """
    out: List[ClassAudit] = []
    for name, total in sorted(total_counts.items(), key=lambda kv: -kv[1]):
        obs = observed_test_counts.get(name) if observed_test_counts else None
        out.append(
            ClassAudit(
                name=name,
                total_instances=total,
                test_instances=obs,
                expected_test=total * test_fraction,
                p_zero_in_test=p_zero_in_split(total, test_fraction),
            )
        )
    return out


def summarize(audits: List[ClassAudit]) -> Dict[str, object]:
    """Aggregate the audit into the numbers that belong in a report."""
    undefined = [a for a in audits if a.verdict == "UNDEFINED"]
    unreliable = [a for a in audits if a.verdict == "UNRELIABLE"]
    ok = [a for a in audits if a.verdict == "OK"]
    n = len(audits)
    return {
        "n_classes_declared": n,
        "n_undefined": len(undefined),
        "n_unreliable": len(unreliable),
        "n_meaningful": len(ok),
        "effective_map_denominator": n - len(undefined),
        "pct_classes_meaningful": (100.0 * len(ok) / n) if n else 0.0,
        "undefined_classes": [a.name for a in undefined],
        "unreliable_classes": [a.name for a in unreliable],
        # Expected number of classes losing all test instances, summed over the
        # per-class probabilities. Linearity of expectation, no independence
        # assumption needed across classes.
        "expected_classes_with_zero_test": sum(a.p_zero_in_test for a in audits),
    }


def ap_delta_noise_floor(audits: List[ClassAudit]) -> float:
    """Smallest mAP change that is not just per-class AP quantization.

    A one-step move in the coarsest reliable class shifts the mean by
    step / n_contributing. Any reported delta below this is inside the grid of
    values the metric can even represent.
    """
    contributing = [a for a in audits if a.verdict != "UNDEFINED"]
    if not contributing:
        return float("nan")
    worst_step = max(a.quantization_step for a in contributing)
    return worst_step / len(contributing)


def format_report(audits: List[ClassAudit]) -> str:
    s = summarize(audits)
    lines = []
    name_w = max(len(a.name) for a in audits)
    lines.append(
        f"{'class'.ljust(name_w)}  {'total':>8}  {'test(exp)':>10}  "
        f"{'AP step':>8}  {'P(0 in test)':>12}  verdict"
    )
    lines.append("-" * (name_w + 56))
    for a in audits:
        step = a.quantization_step
        step_s = "n/a" if a.verdict == "UNDEFINED" else f"{step:.3f}"
        lines.append(
            f"{a.name.ljust(name_w)}  {a.total_instances:>8,}  "
            f"{a.observed_or_expected:>10.1f}  {step_s:>8}  "
            f"{a.p_zero_in_test:>12.3f}  {a.verdict}"
        )
    lines.append("")
    lines.append(f"classes declared              : {s['n_classes_declared']}")
    lines.append(f"AP undefined (dropped by COCO): {s['n_undefined']}  {s['undefined_classes']}")
    lines.append(f"AP unreliable (< {MIN_MEANINGFUL_INSTANCES} instances) : {s['n_unreliable']}  {s['unreliable_classes']}")
    lines.append(f"classes with meaningful AP    : {s['n_meaningful']} "
                 f"({s['pct_classes_meaningful']:.1f}%)")
    lines.append(f"effective mAP denominator     : {s['effective_map_denominator']} "
                 f"(not {s['n_classes_declared']})")
    lines.append(f"mAP noise floor from quantization: "
                 f"{ap_delta_noise_floor(audits):.4f} "
                 f"({100*ap_delta_noise_floor(audits):.2f} mAP points)")
    return "\n".join(lines)
