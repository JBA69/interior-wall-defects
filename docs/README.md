# Interior Wall Defect Detection

Object detection for interior wall defects in a storm-damaged residential property — an
original dataset, a trained YOLO11 model, and an honest account of why it does not yet work.

MAICEN Module 4 Unit 3 · José de Barros Aguiar · Lisbon, 2026

[![Open 01_Training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JBA69/interior-wall-defects/blob/main/notebooks/01_Training.ipynb)
[![Open 02_Inference in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JBA69/interior-wall-defects/blob/main/notebooks/02_Inference.ipynb)

---

## The result, stated first

| | validation | test |
|---|---|---|
| mAP50 | 0.086 | — |
| mAP50-95 | 0.033 | — |
| **defects found** (conf 0.25) | **38 of 363** | **1 of 73** |
| flagged regions that were real | 38 of 144 | 1 of 51 |

**The detector does not work in operational terms.** One true positive across the entire test
split.

The first two rows are Ultralytics' area-under-the-curve figures, which summarise performance
across every confidence threshold at once. The last two are box counts at one threshold,
`conf = 0.25`. Same model, two different questions, and they are kept apart throughout —
§3 of the error analysis reports the first, §5 the second.

That is the headline because it is the truth, and because the useful output of this project is
not the model but the account of why: three controlled training runs, a per-class audit of the
dataset, and a box-by-box analysis of every prediction. Two of the three causes were found by
checking claims that turned out to be wrong.

Full analysis: **[`docs/error_analysis.md`](docs/error_analysis.md)**

---

## The problem

A storm damaged the interior of a Lisbon flat during winter 2025–26. Walls carry cracking,
mould, damp, blistering and peeling paint, in varying combinations, often overlapping.

**Task.** Given a photograph of an interior wall, locate and name visible surface defects.

**Intended use.** Screening support for a building-condition survey — indicating regions a
surveyor should examine.

**Success criteria, set before training:**

| | target | achieved |
|---|---|---|
| Recall prioritised over precision | all classes | ✅ specified, ❌ not delivered |
| Hairline cracks detectable at capture distance | ≥ 0.5 mm at 40 cm | ✅ established by measurement |
| Reproducible by a stranger with no credentials | both notebooks, cold runtime | ✅ verified |
| Usable screening performance | — | ❌ |

---

## Classes

Five classes, defined in full in [`docs/class_definitions.md`](docs/class_definitions.md) —
written and agreed *before* annotation began, so that image 140 is labelled as image 12 was.

| class | definition in one line |
|---|---|
| `crack` | linear fracture in paint or plaster, varying in width along its length |
| `damp_patch` | region darker or differently toned than the wall, soft irregular boundary |
| `mould` | cluster of dark speckling, distinct in texture from the underlying tone |
| `blistering` | paint film lifted from the substrate but **still attached** |
| `peeling_paint` | paint film detached, with **bare substrate visible** |

Key label rules:

- **Tight boxes.** The box touches the feature's pixels on all four sides.
- **Overlapping boxes of different classes are expected.** Mould sits inside damp; both get boxes.
- **Hard negatives get no boxes at all.** 48 frames document 18 confuser types — transfer stains,
  shadow lines, architrave junctions, skirting dust — and carry empty label files on purpose.
- **Label what is visible, not what you know.**

---

## Dataset

**[`interior-wall-defects` v1](https://github.com/JBA69/interior-wall-defects/releases/tag/v1.0)**
· 224 images · 1,866 annotated instances · 1280 px · no augmentation at export

```
https://github.com/JBA69/interior-wall-defects/releases/download/v1.0/interior-wall-defects-v1-yolo11.zip
SHA256  9137b89d2f406adf1fa7dbfc6bf5a2e95db396d39c26a8c80d248a2f0b8b8a42
```

Also published on Roboflow Universe under CC BY 4.0:
[universe.roboflow.com/jose-aguiar-me-com/interior-wall-defects](https://universe.roboflow.com/jose-aguiar-me-com/interior-wall-defects)

### Splits — assigned by room, never at random

| split | rooms | images | instances | background frames |
|---|---|---|---|---|
| train | closet | 134 | 1,430 | 20 (15 %) |
| val | kitchen, hallway | 44 | 363 | 3 (7 %) |
| test | living room, office, bedroom | 46 | 73 | 23 (50 %) |

Each wall surface was photographed up to fifteen times — five distances × three lighting
conditions. A random split would place roughly a dozen near-duplicates of every test frame in
training, and the model would score well by recognising a photograph it had already seen.
Splitting by room makes that structurally impossible.

The cost is that training and evaluation then differ in more than the photographs. That
trade-off, and everything that follows from it, is [`docs/error_analysis.md`](docs/error_analysis.md) §2.

---

## Quick Start

**Click a badge, then Runtime → Run all. Nothing else is needed — no account, no API key, no
upload.** The dataset and the weights are fetched from public release assets and verified by
SHA256 before anything else happens; a mismatch halts the run.

| | notebook | what it does | runtime |
|---|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JBA69/interior-wall-defects/blob/main/notebooks/02_Inference.ipynb) | **[`02_Inference.ipynb`](notebooks/02_Inference.ipynb)** | **Start here.** Downloads and verifies the trained weights, runs inference, accounts for every box as true positive / false negative / false positive / class confusion, and sweeps the confidence threshold. | **CPU, a few minutes** |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JBA69/interior-wall-defects/blob/main/notebooks/01_Training.ipynb) | **[`01_Training.ipynb`](notebooks/01_Training.ipynb)** | Downloads and verifies the dataset, repairs the export's paths, audits it, trains, evaluates, and writes its own provenance record. | **T4 GPU, ~90 min** |

**Start with `02_Inference.ipynb`.** It pulls the published weights from the release, so it has no
dependency on the training notebook and reproduces the reported figures in minutes on a free CPU
runtime.

```
dataset  https://github.com/JBA69/interior-wall-defects/releases/download/v1.0/interior-wall-defects-v1-yolo11.zip
SHA256   9137b89d2f406adf1fa7dbfc6bf5a2e95db396d39c26a8c80d248a2f0b8b8a42

weights  https://github.com/JBA69/interior-wall-defects/releases/download/v1.0/best-run2.pt
SHA256   cc1d5096124f32bc7783d86a817394fc08501ef8291b2fa6c76f9dc1e720fe4f
```

Three training configurations are defined in `01_Training.ipynb` §2. Changing `RUN_TAG`
reproduces any of them.

---

## Results

**Validation is reported as primary.** It selected the checkpoint, so its figures are mildly
optimistic — but the test split holds 73 instances across five classes with `peeling_paint` at
n = 7, which cannot support a per-class measurement. Both tables are published; neither is
hidden.

### Run 2 (the reported model), validation

| class | precision | recall | mAP50 | mAP50-95 | instances |
|---|---|---|---|---|---|
| damp_patch | 0.298 | 0.248 | 0.153 | 0.052 | 101 |
| mould | 0.149 | 0.215 | 0.127 | 0.057 | 93 |
| blistering | 0.236 | 0.147 | 0.107 | 0.043 | 116 |
| crack | 0.044 | 0.087 | 0.040 | 0.013 | 46 |
| peeling_paint | 0.000 | 0.000 | 0.004 | 0.002 | **7** |
| **all** | **0.145** | **0.139** | **0.0859** | **0.0332** | 363 |

`peeling_paint` scoring zero is not a model result. Seven instances cannot produce a meaningful
precision or recall.

### The ablation

| run | change | mAP50 | mAP50-95 | verdict |
|---|---|---|---|---|
| 1 | reduced augmentation, justified on resolution grounds | 0.0649 | 0.0252 | refuted |
| **2** | **augmentation restored** | **0.0859** | **0.0332** | **reported** |
| 2 repeat | *nothing* — same config, same seed, cold runtime | 0.1014 | 0.0385 | the noise floor² |
| 3 | `blistering` ∪ `peeling_paint` → `paint_failure` | 0.1066 | 0.0395 | rejected¹ |

¹ Run 3's mean is computed over four classes, not five. Compared like for like, `paint_failure`
scored 0.102 against `blistering`'s 0.107 — worse. The merge was pre-registered, tested and
rejected.

² Two runs of the identical configuration differ by 8 % of their mean. Averaging them raises the
augmentation effect from the +32 % first reported to **+44 %**, and establishes that any effect
smaller than ±8 % is unmeasurable with one run per condition — which retires several per-class
comparisons elsewhere in the analysis. Run 2 remains the reported model: adopting the better
figures would invalidate every hash in this repository, and pinning is worth more than 0.0155 of
mAP50.

### Recommended operating point

`conf = 0.25` is the library default, not a decision. Re-scoring the same predictions across
thresholds:

**Use `conf = 0.10`** — about a quarter of annotated defects found, at roughly twelve flagged
regions per image, of which about one in six is real. A defensible screening posture for the
stated recall priority. Not a good one.

---

## What went wrong

Three mechanisms, ranked by how much each explains:

1. **Single-room training domain.** Every training instance comes from one closet. 1,430 boxes,
   but many *views* of few *things* — the effective sample size is the number of distinct
   physical defects, an order of magnitude smaller. Validation recall 0.105; test, in unseen
   rooms, 0.014.
2. **Boundary indeterminacy.** mAP50 : mAP50-95 is 2.6 : 1 — boxes land in the right region with
   the wrong edges. The amorphous classes have no crisp boundary, and the annotation inherits
   that.
3. **Under-augmentation** in the initial configuration. Reversed in run 2, worth +44 % on mAP50
   against a measured noise floor of ±8 % — and the localisation losses degrade three to four
   times faster without it, consistently across two runs ([§4.6](docs/error_analysis.md)).

Four things found by checking claims rather than asserting them:

- a resolution argument that survived a physical measurement and then **failed a training test**
- a class confusion reported at 0.14 that proved to be **one box**, amplified by normalisation
- a false positive that turned out to be a **real condition with no class to put it in**
- an argument about which loss generalises worse, withdrawn twice — once because it was read off
  a plot whose axis hid the data, once because **repeating the run reversed the ordering**

Each is documented with the evidence that refuted it:
**[`docs/error_analysis.md`](docs/error_analysis.md)**

---

## Reproducibility checklist

| | |
|---|---|
| Dataset pinned by URL **and** SHA256 | ✅ verified at the start of both notebooks; run halts on mismatch |
| Weights pinned by URL **and** SHA256 | ✅ `02_Inference.ipynb` §4 |
| Library version pinned | ✅ `ultralytics==8.4.166` |
| Torch version pinned | ⚠️ recorded, not pinned — Colab ships it matched to the runtime |
| `seed` fixed, `deterministic=True` | ✅ `seed=0` |
| No credentials used, printed or committed | ✅ both notebooks are keyless by design |
| Training configuration recorded from the object passed to the trainer | ✅ `evidence/run_record.json` |
| Library's own resolved config captured | ✅ `evidence/args.yaml` |
| Cold-runtime *Run All* verified | ✅ both notebooks, 2026-10-02 — inference identical to the digit; training reproduces behaviour, not bytes (§4.6) |
| Class order asserted against the release | ✅ run halts if the export order changes |

### Reproducibility proof

**Verified, not asserted — and the two notebooks reproduce differently.** Both were re-run on
2026-10-02 from deleted Colab runtimes: nothing cached, no credentials, no manual step.

**Inference reproduces exactly.** `02_Inference.ipynb` fetched the dataset and the weights from
the public release, passed both hash checks, and returned an `inference_record.json` identical to
the published one — every count across both splits, to the digit.

**Training reproduces behaviour, not bytes.** `01_Training.ipynb` ran end to end and produced a
model of the same kind, but not the same model: validation mAP50 came back 0.1014 against the
published 0.0859, an 18 % spread. `seed=0` and `deterministic=True` constrain cuDNN kernel
selection; they do not control dataloader worker scheduling or non-deterministic GPU reductions.

That gap is not a defect, and it is the most useful number this project produced — it is the
noise floor every other comparison here is measured against.
[`docs/error_analysis.md`](docs/error_analysis.md) §4.6.

The dataset is a frozen GitHub release asset, not a "latest version" link. The URL says *which
file*; the SHA256 says *which state of that file*. A release asset can be replaced after
publication — the hash is what makes it frozen in fact rather than by convention.

Every figure in this repository traces to a file, and every file to a run. `run_record.json` is
generated from the same dictionary handed to the trainer, so the provenance record cannot
disagree with what executed.

---

## Licences — three independent objects

| object | licence |
|---|---|
| **This repository** (notebooks, documentation, figures) | **MIT** |
| **Dataset** (images + annotations) | **CC BY 4.0** |
| **`best-run2.pt`** (the trained weights) | **AGPL-3.0, inherited** |

**The weights carry a condition the repository does not.** Ultralytics licenses YOLO under
AGPL-3.0 and treats weights trained with it as a derivative work, so anyone placing
`best-run2.pt` behind a commercial service must either release that service's source under
AGPL-3.0 or obtain an Ultralytics Enterprise licence.

That reaches the weights alone. It does not reach this repository, which contains no Ultralytics
code and no weights — the notebooks import the library, and the weights are distributed as a
release asset. It does not reach the dataset either: CC BY 4.0 stands on its own and the images
work with any detector.

Dataset citation:

> Aguiar, J. B. (2026). *Interior Wall Defects v1* [dataset]. CC BY 4.0.
> https://github.com/JBA69/interior-wall-defects

---

## Scope and limits

**This model performs screening, not certification.** It indicates where a surveyor should look.
It does not assess severity, depth, moisture content or structural significance, and its output
is not a condition report.

At the performance measured here it does not meet that bar either. **Absence of a detection
carries no information** — a frame with no boxes is not evidence of a sound wall, and any
workflow treating a clear result as a clear wall misuses this model.

**Operational tempo: Mode 1 — batch.** Images are captured on site and processed afterwards, not
live. Nothing waits on the result, which is what permits `imgsz=1280` and a model sized for
accuracy rather than speed.

Full governance position: [`docs/governance_checklist.md`](docs/governance_checklist.md)

---

## Repository

```
├── README.md
├── LICENSE                            MIT, covering this repository
├── notebooks/
│   ├── 01_Training.ipynb              download, verify, audit, train, evaluate
│   ├── 02_Inference.ipynb             inference, error accounting, threshold sweep
│   └── M4U3_roboflow_1-split_images.ipynb
│                                      split construction (local paths; for transparency,
│                                      not reproducibility)
├── docs/
│   ├── class_definitions.md           the annotation contract, written before annotation
│   ├── capture_protocol.md            how the photographs were taken, and what the grid got wrong
│   ├── capture_resolution_test.md     why 1280, measured before training
│   ├── crack_resolution_test.png      the measurement behind that decision
│   ├── error_analysis.md              what failed, why, and what to fix
│   ├── governance_checklist.md        provenance, privacy, licensing, oversight
│   └── reading_the_outputs.md         how to read the figures, in plain language
├── evidence/
│   ├── results.png                    training curves
│   ├── results_run2a.csv              per-epoch losses and metrics, reported run
│   ├── results_run2b.csv              the same for the replicate — §4.6 is computed from these two
│   ├── confusion_matrix_normalized.png
│   ├── confidence_sweep.png
│   ├── dataset_audit.csv, .png        per-class, per-split instance counts
│   ├── metrics_val.csv, metrics_test.csv
│   ├── per_image_errors.csv           the ranking behind every cited example
│   ├── run_record.json                training provenance, reported run
│   ├── run_record_run2b.json          training provenance, replicate
│   ├── inference_record.json          inference provenance
│   ├── args.yaml                      Ultralytics' own resolved configuration
│   └── overlays/
│       ├── fn-1-kitchen-c-c20.jpg     false negatives
│       ├── fp-1-hallway-a-c40.jpg     a false positive with no class to put it in
│       └── cc-1-office-c-c40.jpg      class confusion
└── data/
    └── areas.csv                      the per-frame capture log, 244 rows
```

---

## Disclaimer

> **This model is an assistive tool for preliminary screening only. It produces False Negatives.
> It must NOT be used as the sole verifier for life-safety decisions.**

The dataset and model in this repository were produced for a master's assignment. They are
published as a method and a baseline, not as a tool. Every output requires review by a qualified
person, who carries the professional responsibility for any assessment made from it.

Full governance position, including the measured conditions under which this model must not be
used: [`docs/governance_checklist.md`](docs/governance_checklist.md)
