#!/usr/bin/env python3
"""Long-tail evaluation audit for the 31-class dental panoramic dataset.

Runs end to end with no dataset download required, because it operates on the
split sizes and per-class counts the dataset spec publishes. Point it at real
annotation files with --coco or --yolo to run the same audit on the actual data.

    python main.py                          # audit from published spec
    python main.py --coco path/to/_annotations.coco.json
    python main.py --yolo path/to/labels/train
    python main.py --demo-boundary          # Boundary IoU vs mask IoU
    python main.py --demo-rfs               # repeat-factor sampling
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from longtail import client_spec as spec
from longtail import distribution, rfs, split_audit
from longtail.boundary_iou import boundary_iou, mask_iou

SEED = 1337


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def audit_from_spec() -> None:
    hr("1. WHAT THE SPEC ACTUALLY PUBLISHES")
    print(f"  images      : {spec.TOTAL_IMAGES:,}  "
          f"(train {spec.SPLITS['train']['images']:,} / "
          f"valid {spec.SPLITS['valid']['images']:,} / "
          f"test {spec.SPLITS['test']['images']:,})")
    print(f"  annotations : {spec.TOTAL_ANNOTATIONS:,}")
    print(f"  test share  : {100*spec.TEST_FRACTION:.2f}% of annotations")
    print(f"  classes     : {len(spec.CLASS_NAMES)}")
    print(f"  coverage    : {spec.coverage_note()}")
    print(f"  stated imbalance ratio : {spec.stated_imbalance_ratio():,.0f} : 1")
    print("\n  For reference, LVIS -- the benchmark the long-tail detection")
    print("  literature is built on -- has an imbalance ratio near 1,000:1.")
    print("  This dataset, as described, is roughly 33x more skewed than that.")

    hr("2. CAN A TEST-SET AP EXIST FOR THE TAIL CLASSES?")
    print("  Reading A: the quoted counts are whole-dataset totals.")
    audits = split_audit.audit(
        {k: v for k, v in spec.PUBLISHED_COUNTS.items() if v is not None},
        test_fraction=spec.TEST_FRACTION,
    )
    print()
    for a in audits:
        print(f"    {a.name:<16} total={a.total_instances:>3}  "
              f"expected in test={a.expected_test:>5.2f}  "
              f"P(zero in test)={a.p_zero_in_test:>6.1%}  -> {a.verdict}")
    exp_zero = sum(a.p_zero_in_test for a in audits)
    print(f"\n    Expected number of these {len(audits)} classes with NO test "
          f"instances: {exp_zero:.2f}")
    print("    Those classes return AP = -1 from COCOeval and are dropped from")
    print("    the mean. The headline '31-class mAP' is then not over 31 classes.")

    print("\n  Reading B: the quoted counts are train-split only.")
    print("    Then the test split holds separate instances of these classes,")
    print("    drawn from an even smaller pool. Strictly worse. Either reading")
    print("    gives the same conclusion, so the ambiguity does not need")
    print("    resolving before the risk is real.")

    hr("3. WHAT SIZE OF IMPROVEMENT WOULD EVEN BE MEASURABLE?")
    print("  With k ground-truth instances in test, recall takes only k+1 values,")
    print("  so per-class AP is quantized in steps of about 1/k:\n")
    for a in audits:
        k = a.observed_or_expected
        if k >= 1:
            print(f"    {a.name:<16} k={k:.2f}  ->  AP resolution {1/k:.2f} "
                  f"({100/k:.0f} AP points per single detection)")
        else:
            print(f"    {a.name:<16} k={k:.2f}  ->  expected below 1 instance; "
                  f"AP most likely undefined")
    print("\n  A class at k=1 is effectively a coin flip: one detection moves its")
    print("  AP the full distance from 0 to 1. A reported '+2 mAP' driven by such")
    print("  a class is noise. This is why the contract should be written against")
    print("  a stratified metric, not a single pooled number.")

    hr("4. THE FIX: REPORT HEAD / MID / TAIL SEPARATELY")
    print("  LVIS convention (Gupta et al., CVPR 2019, arXiv:1904.03797):")
    print("    rare     : present in  < 10 training images")
    print("    common   : present in 10-100 training images")
    print("    frequent : present in  > 100 training images")
    print("\n  Report AP_rare, AP_common, AP_frequent separately, plus the count")
    print("  of classes actually contributing to each. A single 31-class mAP on")
    print("  this dataset is dominated by the 8,000-33,000 instance classes")
    print("  (Filling, impacted tooth, Root Canal Treatment) and can rise while")
    print("  every tail class gets worse.")


def audit_from_coco(path: str) -> None:
    hr(f"COCO AUDIT: {path}")
    stats = distribution.from_coco(path, split="from-file")
    print(f"  images {stats.n_images:,} | annotations {stats.n_instances:,} | "
          f"classes {len(stats.classes)}")
    print(f"  imbalance ratio: {stats.imbalance_ratio:,.0f} : 1\n")
    print(distribution.format_table(stats))
    empty = stats.empty_classes()
    if empty:
        print(f"\n  ZERO-INSTANCE CLASSES ({len(empty)}): "
              f"{[c.name for c in empty]}")
        print("  These return AP = -1 and are silently excluded from mAP.")
    groups = stats.by_group()
    print(f"\n  rare {len(groups['rare'])} | common {len(groups['common'])} | "
          f"frequent {len(groups['frequent'])}")


def audit_from_yolo(path: str) -> None:
    hr(f"YOLO AUDIT: {path}")
    stats = distribution.from_yolo(path, spec.CLASS_NAMES, split="from-file")
    print(f"  label files {stats.n_images:,} | instances {stats.n_instances:,}")
    print(f"  imbalance ratio: {stats.imbalance_ratio:,.0f} : 1\n")
    print(distribution.format_table(stats))


def demo_boundary() -> None:
    hr("BOUNDARY IoU vs MASK IoU (Cheng et al., CVPR 2021, arXiv:2103.16562)")
    rng = np.random.default_rng(SEED)
    h = w = 400

    # A large elongated structure, the shape class that mask IoU flatters most:
    # think Mandibular Canal or maxillary sinus rather than a small caries lesion.
    yy, xx = np.mgrid[0:h, 0:w]
    gt = ((yy > 150) & (yy < 250) & (xx > 40) & (xx < 360))

    print("  Ground truth: a 320x100 elongated structure (canal-like).\n")
    print(f"  {'prediction':<34}{'mask IoU':>10}{'boundary IoU':>15}")
    print("  " + "-" * 58)

    # 1. Perfect.
    print(f"  {'exact match':<34}{mask_iou(gt, gt):>10.3f}"
          f"{boundary_iou(gt, gt):>15.3f}")

    # 2. Uniformly eroded by 6px: a visibly thinner canal, small interior loss.
    pred = ((yy > 156) & (yy < 244) & (xx > 46) & (xx < 354))
    print(f"  {'6px thinner all round':<34}{mask_iou(gt, pred):>10.3f}"
          f"{boundary_iou(gt, pred):>15.3f}")

    # 3. Ragged border: correct on average, wrong everywhere locally.
    noise = rng.random((h, w)) < 0.5
    band = gt & ~((yy > 158) & (yy < 242) & (xx > 48) & (xx < 352))
    pred = np.where(band, noise, gt)
    print(f"  {'ragged/noisy border':<34}{mask_iou(gt, pred):>10.3f}"
          f"{boundary_iou(gt, pred):>15.3f}")

    print("\n  Read the gap between the two columns. Mask IoU stays high because")
    print("  the interior dominates the pixel count; Boundary IoU drops because")
    print("  the border is what changed. On the large anatomical classes here,")
    print("  a model can gain mask mAP while its tracings get visibly worse to")
    print("  the clinician reading them. Report both.")


def demo_rfs() -> None:
    hr("REPEAT FACTOR SAMPLING (Gupta et al., CVPR 2019, arXiv:1904.03797)")
    rng = random.Random(SEED)

    # Synthetic corpus with this dataset's shape: 3 head classes in most images,
    # a mid group, and 3 tail classes appearing a handful of times.
    n_images = 10_000
    image_classes = []
    for i in range(n_images):
        cs = set()
        for c in (0, 1, 2):                      # head, near-ubiquitous
            if rng.random() < 0.75:
                cs.add(c)
        for c in range(3, 12):                   # mid
            if rng.random() < 0.06:
                cs.add(c)
        for c, n in ((12, 9), (13, 4), (14, 1)):  # tail, fixed occurrence counts
            if i < n:
                cs.add(c)
        image_classes.append(cs or {0})

    freqs = rfs.class_image_frequencies(image_classes)
    r_c = rfs.class_repeat_factors(freqs, threshold=0.001)
    r_i = rfs.image_repeat_factors(image_classes, threshold=0.001)

    print(f"  corpus: {n_images:,} images\n")
    print(f"  {'class':<8}{'f(c)':>10}{'r(c)':>10}   role")
    print("  " + "-" * 42)
    for c, role in ((0, "head"), (5, "mid"), (12, "tail n=9"),
                    (13, "tail n=4"), (14, "tail n=1")):
        print(f"  {c:<8}{freqs.get(c, 0):>10.5f}{r_c.get(c, 1.0):>10.2f}   {role}")

    epoch = rfs.sample_epoch_indices(r_i, rng)
    print(f"\n  expected epoch size : {rfs.expected_epoch_size(r_i):,.0f} "
          f"(vs {n_images:,} without RFS)")
    print(f"  sampled epoch size  : {len(epoch):,}")

    base = sum(1 for cs in image_classes if 14 in cs)
    after = sum(1 for i in epoch if 14 in image_classes[i])
    print(f"\n  class 14 (n=1) draws per epoch: {base} -> {after} "
          f"({after/max(base,1):.0f}x)")
    print("\n  RFS costs nothing but sampler configuration and is the base every")
    print("  stronger method (EQLv2, Seesaw, ECM) is reported on top of. It is")
    print("  the correct first experiment, not a novel contribution.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", help="path to a COCO annotations JSON")
    p.add_argument("--yolo", help="path to a directory of YOLO label .txt files")
    p.add_argument("--demo-boundary", action="store_true")
    p.add_argument("--demo-rfs", action="store_true")
    args = p.parse_args(argv)

    random.seed(SEED)
    np.random.seed(SEED)

    ran_any = False
    if args.coco:
        audit_from_coco(args.coco); ran_any = True
    if args.yolo:
        audit_from_yolo(args.yolo); ran_any = True
    if args.demo_boundary:
        demo_boundary(); ran_any = True
    if args.demo_rfs:
        demo_rfs(); ran_any = True

    if not ran_any:
        audit_from_spec()
        demo_boundary()
        demo_rfs()

    hr()
    print("Seeded with SEED=%d. Every number above is reproducible." % SEED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
