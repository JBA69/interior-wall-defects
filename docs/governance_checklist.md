# AECO Governance Checklist

**Project:** `interior-wall-defects` · MAICEN Module 4 Unit 3
**Author:** José de Barros Aguiar
**Dataset:** v1 · 224 images · 1,866 instances · SHA256 `9137b89d2f40…`
**Model:** YOLO11s, run 2 · SHA256 `cc1d5096124f…`
**Last reviewed:** 2026-10-03

> **This model is an assistive tool for preliminary screening only. It produces False Negatives.
> It must NOT be used as the sole verifier for life-safety decisions.**

---

## 1. Data Provenance

- **Source:** Original photographs taken by the author on an iPhone 12 Pro, in a storm-damaged
  residential interior in Lisbon that he occupied at the time of capture. Nothing is scraped,
  purchased, inherited from another dataset, or generated.
- **Date Collection:** 14–20 September 2026, across five sessions.
- **Owner:** The author, who took every frame and holds all rights to them.

### Capture sessions

| date | session | rooms |
|---|---|---|
| 14 Sep 2026 | defect capture | closet |
| 16 Sep 2026 | hard negatives — confusers | all six rooms |
| 17 Sep 2026 | hard negatives — sound wall | bedroom, living room, kitchen |
| 18 Sep 2026 | defect capture | hallway, kitchen, living room, office |
| 20 Sep 2026 | holdout frames | hallway, kitchen, living room |

**No client or project material.** No image originates from professional work. The brief excluded
hospitals and laboratories; this dataset contains neither, and no commercial or confidential
setting of any kind.

**Third-party interests.** The property was storm-damaged and under repair, with the author
moving out. No insurance claim depends on these photographs and no third party has an interest in
them. The landlord was not a party to the dataset.

**Chain of custody.** Raw captures → local room-structured archive (`data/areas.csv`, 244 logged
frames) → selected subset uploaded to Roboflow → annotated → exported → published as a frozen
GitHub release asset, pinned by SHA256. Every step is recorded, and
`notebooks/01_Training.ipynb` §6b cross-checks the published export against the capture log and
names any discrepancy.

**What is published and what is not.** The annotated 1280 px export is public: 224 images, of
which 46 are background frames carrying empty label files on purpose. The full raw archive —
roughly 700 frames across six rooms, including deliberately excluded and duplicate captures — is
not published. Twelve holdout frames are retained unpublished.

**Credentials.** No API key appears in any committed cell, any cell output, or the git history.
The dataset and weights are fetched from public release assets over plain HTTPS; no
authentication is used or needed. Had a key ever been committed, deleting it in a later commit
would not be sufficient — it would have to be revoked in Roboflow and reissued.

---

## 2. PII Handling (Privacy)

- **Faces / License Plates:** **None present.** No person appears in any frame, in any capture
  session. No vehicles and no registration plates appear in any frame. The subject of every frame
  is an interior wall surface.
- **Protection Strategy:** **No PII present in the dataset, so no anonymisation was applied.**
  Face-blurring preprocessing was considered and is not applicable: there is nothing to blur.
  Roboflow's blur augmentation was therefore not enabled.

**Identifying content, reviewed frame by frame.** Frames were reviewed for anything that could
identify the property or its occupants: no documents, no screens, no correspondence, no personal
possessions with names on them, no exterior views showing street numbers or neighbouring
properties.

**Location.** Described only as a residential interior in Lisbon. No address, building name, or
precise location is published, and EXIF GPS is not present in the exported images — Roboflow's
export strips metadata.

**Consent.** The author is the occupant and the photographer. No third-party consent was
required, and none is implied by publication.

**Withdrawal.** Because the data concerns a property rather than a person, there is no data
subject with a right of erasure to exercise. Should that assessment prove wrong, the release
asset can be withdrawn and the Roboflow dataset unpublished; the SHA256 in this repository would
then identify exactly which bytes were removed.

---

## 3. Risk Statement

**Intended use.** Screening support for a building-condition survey: the model indicates regions
of an interior wall that a surveyor may wish to examine. Batch processing after a site visit,
with no real-time constraint.

- **High-Impact False Negative:** **A missed defect is a defect behind a wall somebody is about
  to sign for.** At the default threshold the model finds **38 of 363** annotated defects in
  validation and **1 of 73** in the held-out test rooms — it misses roughly nine in ten. In a
  condition survey the consequence is remediation deferred until the substrate is affected:
  damp progressing behind intact-looking paint, or mould establishing in a cold corner, each
  discovered a season later at several times the cost. In a tenancy or a sale it becomes a
  dispute about who knew what and when.

  **The operational consequence that matters most: absence of a detection carries no
  information.** A frame with no boxes is not evidence of a sound wall. Any workflow that treats
  a clear result as a clear wall misuses this model.

- **High-Impact False Positive:** **A flagged region that is sound sends a surveyor, or a
  contractor, to open up a wall that did not need opening.** 106 false positives in validation
  and 50 in test. The direct cost is a wasted inspection; the larger cost is that a tool which
  cries wolf gets ignored, and the real detections go with it.

  One complication, documented in `docs/error_analysis.md` §5.5: figure FP-1 is a prediction on
  a frame annotated with no boxes, at an architrave junction that carries a real mark, decades of
  accumulated paint build-up, and movement cracking at the skirting transition. None of that is
  one of the five classes, so the annotation contract correctly says no box — and the model,
  finding something real and holding only five names, assigned the nearest. **An unknown share
  of the 156 false positives may be detections of real but out-of-scope conditions.** The
  precision figures treat both identically.

**The asymmetry was specified and is not delivered.** All five classes were defined
recall-priority in `docs/class_definitions.md`, because a missed defect costs more than a false
alarm. At `conf = 0.25` the model returns precision 0.264 against recall 0.105 — the opposite
balance. `conf = 0.10` is recommended instead: about a quarter of annotated defects found, at
roughly twelve flagged regions per image of which about one in six is real. A defensible
screening posture for the stated priority. Not a good one.

### Do not use this model under these conditions

Stated as measured thresholds rather than general cautions.

| condition | do not use | evidence |
|---|---|---|
| **Standoff beyond 40 cm** | Capture at 20 cm; never beyond 40 | Validation recall 0.132 at 20 cm, 0.085 at 40 cm, 0.059 at 80 cm, **0.000 at 100 cm** |
| **Cracks below 0.5 mm apparent width** | Not detectable at this capture distance and resolution | Measured in `docs/capture_resolution_test.md` before training; confirmed afterwards by the fall in recall with distance |
| **Artificial light, or after dark** | Untested, not merely degraded | Evaluation was 91–100 % daylight while 40 % of training was artificial light; the two cannot be separated from room |
| **Oblique or wide context framings** | Outside the training domain | No context-framed and no 80 cm frames in training, yet 44 of 436 evaluation instances (10 %) sit in those framings |
| **Any building other than a residential painted interior** | Out of scope and unmeasured | Exterior surfaces, non-painted finishes and other building types were never captured |
| **Per-class test figures as a performance estimate** | Not a measurement | 73 test instances across five classes, `peeling_paint` at n = 7; two runs of the identical configuration scored 0.0134 and 0.0423 on it |

### Known biases, measured rather than asserted

From `docs/error_analysis.md`:

| bias | evidence |
|---|---|
| Single-room, single-building training domain | all 1,430 training instances from one closet; validation recall 0.105 against test 0.014 |
| Lighting imbalance | training 35 % daylight, evaluation 91–100 % daylight; lighting confounded with room, so the effect cannot be isolated |
| Distance dependence | recall 0.132 at 20 cm falling to 0.000 at 100 cm |
| Majority-class collapse | `damp_patch` → `mould` is the largest confusion at 11 instances; `mould` is 44 % of training boxes |
| Unmeasurable class | `peeling_paint` has 7 validation instances; its reported 0.000 precision is a measurement failure, not a model result |
| Framing gap | no context-framed images in training; 27 evaluated instances are in that framing |
| Annotation ceiling unmeasured | every figure scores against one annotator's single pass; inter-annotator agreement was never measured, so the metric's real maximum is unknown |

---

## 4. Human-in-the-Loop

- **Review Process:** **Every output is reviewed by a qualified person before it informs
  anything.** The model produces candidate regions and nothing else: it makes no decision,
  triggers no action, and writes to no record. A surveyor verifies every detection, and —
  because absence of a detection carries no information — inspects the wall independently of
  the model's output rather than only where it has drawn a box.

**Screening, not certification.** The distinction is the whole of the model's permitted role:

| | |
|---|---|
| **Screening — allowed** | "This model detects potential defects for a human to inspect." |
| **Certification — forbidden** | "This model certifies the wall is sound." |

The model reports what is visible and where. A human assesses what it means, and carries the
professional responsibility for that assessment. At the performance measured here the model does
not meet the bar for the screening role either; it is published as a baseline and a method, not
as a deployable tool.

**Responsible party.** The author, for the dataset, the training procedure, the documentation and
the published claims.

**Redress.** Anyone finding an error in the dataset, the documentation or the claims can open an
issue on the repository. Corrections will be made in place and the affected documents dated.

---

## 5. License

- **Type:** **MIT** — for this repository: the notebooks, documentation and figures.

Three separate objects carry three separate licences. Conflating them is the common error.

| object | licence | what it permits and requires |
|---|---|---|
| **This repository** (notebooks, documentation, figures) | **MIT** | Anyone may use, modify, redistribute or sell it, with the copyright notice retained. No warranty, no liability. |
| **The dataset** (images + annotations, on Roboflow Universe and as a release asset) | **CC BY 4.0** | Anyone may use, modify and redistribute, including commercially, provided the author is credited. The same licence applies on Universe and on the release asset. |
| **`best-run2.pt`** (the trained weights) | **AGPL-3.0, inherited** | Not a choice. See below. |

**The Ultralytics condition, stated plainly.** Ultralytics licenses YOLO under AGPL-3.0, and
Ultralytics' own position is that weights trained with it are a derivative work. Anyone who takes
`best-run2.pt` and puts it behind a commercial service must either release that service's source
under AGPL-3.0 or obtain an Ultralytics Enterprise licence.

This does not affect the repository, which contains no Ultralytics code and no weights — the
notebooks import the library, and the weights are distributed as a release asset. It does not
affect the dataset either: CC BY 4.0 stands on its own and the images work with any detector. It
affects the **weights**, and it is stated here because a reader could otherwise assume the
repository's licence is the only one that matters.

**Attribution for reuse of the dataset:**

> Aguiar, J. B. (2026). *Interior Wall Defects v1* [dataset]. CC BY 4.0.
> https://github.com/JBA69/interior-wall-defects

---

## 6. Reproducibility, and its limits

Both notebooks run from a cold runtime with no credentials. Dataset and weights are pinned by URL
and SHA256; Ultralytics is pinned to 8.4.166; `seed=0`, `deterministic=True`. Both were verified
from deleted Colab runtimes on 2026-10-02. Three constraints remain, stated rather than hidden:

- **Training is not bit-reproducible, and `deterministic=True` does not make it so.** Re-running
  `01_Training.ipynb` under the identical configuration and seed returned validation mAP50 of
  0.1014 against the published 0.0859 — an 18 % spread. The flag constrains cuDNN kernel
  selection; it does not control dataloader worker scheduling or non-deterministic GPU
  reductions. What this repository claims is that the notebook runs end to end from a cold
  runtime and produces a model of the same kind, not that it recreates `best-run2.pt`. Inference
  *is* exactly reproducible: `02_Inference.ipynb` returned identical counts on both splits, to
  the digit. The spread is reported as a finding rather than a defect — it is the noise floor
  against which every comparison in `error_analysis.md` §4 is judged, and it retired several
  claims that had rested on single runs. See §4.6 there.
- **Torch is not pinned.** Colab ships it matched to the runtime's CUDA build and pinning would
  force a multi-gigabyte reinstall on every run. The version actually used is recorded in
  `run_record.json`. Pin what you control; record what you don't. Two separate cold runtimes both
  drew `torch 2.11.0+cpu`, which narrows the practical exposure without removing it.
- **Training needs a GPU**, and a free Colab account exhausts its allowance in roughly three runs
  at `imgsz=1280`. Inference runs on CPU in minutes.

---

## 7. Errors found and corrected during the work

Recorded because a checklist that reports only successes is not a checklist.

- An augmentation configuration justified on resolution grounds was refuted by experiment and
  reversed (`error_analysis.md` §4.2).
- A class merge was pre-registered, tested and rejected (§4.3).
- A class confusion initially reported as significant proved to be a single instance amplified by
  column normalisation (§4.3).
- A colour-cast hypothesis was abandoned after checking the raw photographs.
- An argument about which validation loss generalises worse was withdrawn twice: once because it
  had been read off a plot whose y-axis was scaled to the warm-up spike and hid the data, and once
  because repeating the run reversed the ordering of the loss minima (§4.1, §4.6).
- **A hash check that silently did not run.** `MODEL_SHA256` was left empty in
  `02_Inference.ipynb`, and the verification is guarded by `if MODEL_SHA256.strip():` — so the
  weights were never checked, while this checklist and the README both stated they were. Found
  while preparing the cold-runtime verification, before that run could certify the false claim.
  A conditional check with an unset expectation passes silently; that is the failure mode worth
  remembering from this project.
- A background-frame count of 48 was carried in the documentation; counting the empty label files
  in the published export gives **46**. The 48 was an archive-side figure.
- Three logged training images never reached the export; identified by the log-versus-export
  cross-check and marked `exclude=yes` rather than quietly dropped.

---

## 8. Traceability

Every reported figure traces to a file, and every file to a run.

| claim | evidence |
|---|---|
| dataset identity | `DATASET_SHA256` verified at the start of both notebooks; the run halts on mismatch |
| model identity | `MODEL_SHA256` verified in `02_Inference.ipynb` §4; the run halts on mismatch |
| training configuration | `evidence/run_record.json`, generated from the same dictionary passed to the trainer, plus Ultralytics' own `evidence/args.yaml` |
| per-epoch losses and metrics | `evidence/results_run2a.csv`, `evidence/results_run2b.csv` |
| metrics | `evidence/metrics_val.csv`, `evidence/metrics_test.csv` |
| dataset composition | `evidence/dataset_audit.csv` |
| which examples were cited and why | ranked in `evidence/per_image_errors.csv`, not selected by eye |
| inference configuration | `evidence/inference_record.json` |

---

## Checklist

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | All images owned by the author | ✅ | §1 |
| 2 | No client or project material | ✅ | §1 |
| 3 | Capture dates recorded | ✅ | §1 |
| 4 | No faces, no plates, no people | ✅ | §2 |
| 5 | Protection strategy stated, with reasons | ✅ | §2 — no PII present, so no blurring applied |
| 6 | Location not disclosed beyond city | ✅ | §2 |
| 7 | No credentials in code, output or git history | ✅ | §1; both notebooks are keyless |
| 8 | High-impact false negative stated | ✅ | §3 |
| 9 | High-impact false positive stated | ✅ | §3 |
| 10 | Do-not-use conditions stated as measured thresholds | ✅ | §3 table |
| 11 | Performance reported without softening | ✅ | §3; `error_analysis.md` §1 |
| 12 | Known biases measured, not asserted | ✅ | §3 table; `error_analysis.md` §2 |
| 13 | Human oversight required and specified | ✅ | §4 |
| 14 | Screening-versus-certification distinction stated | ✅ | §4 |
| 15 | Required disclaimer present in the README | ✅ | README; repeated at the head of this file |
| 16 | Repository licence stated | ✅ | §5 — MIT |
| 17 | Dataset licence stated, matching Roboflow Universe | ✅ | §5 — CC BY 4.0 |
| 18 | Dependency licence and its reach stated | ✅ | §5 — Ultralytics AGPL-3.0 extends to trained weights |
| 19 | Dataset pinned by hash | ✅ | both notebooks |
| 20 | Model pinned by hash | ✅ | `02_Inference.ipynb` §4 |
| 21 | Library version pinned | ✅ | `ultralytics==8.4.166` |
| 22 | Torch version pinned | ⚠️ | recorded, not pinned — §6, with reason |
| 23 | Cold-runtime *Run All* verified | ✅ | both notebooks, 2026-10-02 — inference identical to the digit; training reproduces behaviour, not bytes (§6) |
| 24 | Three failure cases identified and explained | ✅ | `error_analysis.md` §5.5 — FN-1, FP-1, CC-1 |
| 25 | Holdout frames evaluated | ⬜ | `02_Inference.ipynb` §9 — not run; stated as open rather than omitted |
| 26 | Errors and reversals documented | ✅ | §7; `error_analysis.md` §4 |

⬜ = outstanding at the time of writing. ⚠️ = partially met, with the reason stated.
