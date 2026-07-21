# dental-longtail-eval

An evaluation audit for severely long-tailed detection and instance segmentation
on dental panoramic radiographs.

This is not a model. It is the harness that decides whether a model's reported
improvement is real, which on a dataset this skewed is the harder half of the
problem.

## Why this exists

The dataset it targets has 31 classes and, by its own specification, an
imbalance ratio near **33,000 : 1** (8,000-33,000 instances for `Filling`,
`impacted tooth` and `Root Canal Treatment`; 9, 4 and 1 instance for
`Fracture teeth`, `TAD` and `bone defect`). For scale, LVIS, the benchmark the
entire long-tail detection literature is built on, sits near 1,000 : 1.

At that skew two things break quietly, and both break in the direction of
making a model look better than it is.

### 1. Part of the mAP denominator disappears

`COCOeval` returns `AP = -1` for any category with no ground-truth instances in
the evaluation set, and those entries are excluded from the mean. A headline
"31-class mAP" computed on a split where several classes have no test instances
is silently a 25-class mAP. The number is not incorrect; it is answering a
different question than the one it appears to answer.

Running the audit on the published split (10.93% of annotations held out as
test):

| class | total instances | expected in test | P(zero in test) |
|---|---|---|---|
| `Fracture teeth` | 9 | 0.98 | 35.3% |
| `TAD` | 4 | 0.44 | 63.0% |
| `bone defect` | 1 | 0.11 | 89.1% |

Expected number of those three classes with no test instances at all: **1.87**.

The probability model treats instances as independently assigned to the split,
which is *optimistic*. Real instances cluster within images, which makes an
all-or-nothing outcome more likely, not less. These are lower bounds on the risk.

### 2. Tail AP is quantized far coarser than the improvements being claimed

With `k` ground-truth instances in the test split, *recall* can take only the
`k+1` values `{0, 1/k, ..., 1}`. AP is not a clean multiple of `1/k`, because
precision still varies with false positives, but it inherits that coarseness:
per-class AP moves in jumps on the order of `1/k` rather than continuously.

At `k = 1` a class AP is close to a coin flip: a single detection moves it the
whole distance from 0 to 1. A reported "+2 mAP" driven by such a class is noise,
not a result.

`split_audit.ap_delta_noise_floor()` computes the smallest mAP change that is
not pure quantization, which is the number a performance contract should be
written against.

### 3. Mask IoU cannot see the errors a clinician sees

The large anatomical classes here (`Mandibular Canal`, `maxillary sinus`,
`Bone Loss`) are judged by their border, and mask IoU is least sensitive to
border error exactly when objects are large, because interior pixels dominate
the count.

Measured by this repo on a synthetic canal-shaped structure (`--demo-boundary`):

| prediction | mask IoU | Boundary IoU |
|---|---|---|
| exact match | 1.000 | 1.000 |
| 6px thinner all round | 0.846 | **0.294** |
| ragged / noisy border | 0.900 | **0.395** |

A model can gain mask mAP while its canal tracings get visibly worse to the
person reading them. Report both metrics, not one.

## Install and run

```bash
pip install -r requirements.txt
python main.py                    # full audit from the published spec, no download needed
python main.py --demo-boundary    # Boundary IoU vs mask IoU
python main.py --demo-rfs         # repeat factor sampling
```

Point it at real annotations to run the same audit on actual data:

```bash
python main.py --coco path/to/_annotations.coco.json
python main.py --yolo path/to/labels/train
```

The COCO path reports per-class instance and image counts, the imbalance ratio,
the LVIS frequency grouping, and an explicit list of zero-instance classes. The
YOLO path handles both box rows (5 fields) and polygon rows (1 + 2N fields).

## Modules

| Module | What it does |
|---|---|
| `longtail/distribution.py` | Per-class instance and image counts from COCO JSON or YOLO labels; LVIS rare/common/frequent grouping; zero-instance detection |
| `longtail/split_audit.py` | Whether a per-class AP is defined, and whether it is reliable enough to report; mAP noise floor from quantization |
| `longtail/boundary_iou.py` | Boundary IoU (Cheng et al., CVPR 2021) with a dependency-free erosion fallback |
| `longtail/rfs.py` | Repeat factor sampling (Gupta et al., CVPR 2019), the baseline every stronger long-tail method is reported on top of |
| `longtail/client_spec.py` | Verified dataset facts, each traced to a source, with unpublished values recorded as `None` rather than estimated |

## A note on `client_spec.py`

Only 6 of 31 class counts are published (3 exactly, 3 as a range). The remaining
25 are recorded as unknown rather than filled with plausible estimates, because
the unstated tail counts are precisely where the evaluation breaks. Inventing
them would defeat the purpose of the audit.

One discrepancy is recorded rather than smoothed over: the specification's split
sizes sum to 13,932 images while the source Roboflow project lists 13,814. The
two are not expected to match, since the working copy is a re-export.

## What to do about it

Ranked by evidence strength per unit of effort, all citable:

1. **Repeat factor sampling** (Gupta et al., CVPR 2019, [arXiv:1904.03797](https://arxiv.org/abs/1904.03797)).
   Sampler configuration only. The base every method below is measured on top of.
2. **Copy-paste augmentation** (Ghiasi et al., CVPR 2021, [arXiv:2012.07177](https://arxiv.org/abs/2012.07177)).
   Their copy-paste model beat the LVIS 2020 Challenge winning entry by +3.6
   mask AP on rare categories (abstract, verbatim). Additive with RFS, and it
   works natively in YOLOv8-seg via the `copy_paste` hyperparameter because
   polygon labels already exist.
3. **ECM Loss** (Cho & Krähenbühl, ECCV 2022, [arXiv:2301.09724](https://arxiv.org/abs/2301.09724)).
   Best rare-class AP in the published LVIS comparison (AP_r 19.5 vs 9.5 for the
   RFS + cross-entropy baseline), hyperparameter-free, drop-in sigmoid replacement.
4. **Seesaw Loss** (Wang et al., CVPR 2021, [arXiv:2008.10032](https://arxiv.org/abs/2008.10032))
   and **Federated Loss** (Zhou et al., [arXiv:2103.07461](https://arxiv.org/abs/2103.07461))
   as the mature alternatives with MMDetection and Detectron2 implementations.
5. **Class consolidation.** Some classes will not train at any loss function.
   Merging semantically adjacent labels is the honest path, and there is
   precedent: a 2026 *Diagnostics* study consolidated 93 classes to 35 and cut
   imbalance from 2560:1 to 61:1.

One caution from the same literature: **LWS actively hurts detection** (AP_r 2.0
against a 9.5 baseline) even though decoupled classifier rebalancing works well
for classification. Techniques do not port from ImageNet-LT to detection
unexamined.

## Reproducibility

Everything is seeded (`SEED = 1337`) and runs on CPU in a few seconds. No
dataset download is required for the audit, because it operates on the published
split sizes and class counts.

## References

- Gupta, Dollár & Girshick. *LVIS: A Dataset for Large Vocabulary Instance Segmentation.* CVPR 2019. [arXiv:1904.03797](https://arxiv.org/abs/1904.03797)
- Cheng, Girshick, Dollár, Berg & Kirillov. *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation.* CVPR 2021. [arXiv:2103.16562](https://arxiv.org/abs/2103.16562)
- Ghiasi et al. *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation.* CVPR 2021. [arXiv:2012.07177](https://arxiv.org/abs/2012.07177)
- Cho & Krähenbühl. *Long-tail Detection with Effective Class-Margins.* ECCV 2022. [arXiv:2301.09724](https://arxiv.org/abs/2301.09724)
- Wang et al. *Seesaw Loss for Long-Tailed Instance Segmentation.* CVPR 2021. [arXiv:2008.10032](https://arxiv.org/abs/2008.10032)

## License

MIT
