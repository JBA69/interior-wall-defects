# Class Definitions — Interior Wall Defects

**Status: draft for review.** Every row needs your eye on it before annotation starts.
The rows marked **DECIDE** are the ones where I've made a call you may disagree with.

Purpose: a written specification of what each class means, agreed before any box is
drawn, so that image 140 is labelled the same way as image 12. This document is the
authority during annotation — if a decision isn't in here, stop and add it rather than
deciding on the spot.

---

## Capture context these rules assume

| | |
|---|---|
| Camera | iPhone 12 Pro, main camera (26 mm equiv.), 3024 × 4032 |
| Standard distance | ~40 cm (`c40`), frame covering ≈ 50 cm across the short side |
| Training resolution | `imgsz=1280` |
| Effective resolution | ≈ 23 px/cm at inference — **1 px ≈ 0.45 mm** |

Minimum sizes below are stated in **millimetres at 40 cm capture distance**, because
that is what you can judge while annotating. The pixel equivalent at `imgsz=1280`
is given alongside.

Resolution basis: `docs/capture_resolution_test.md`.

---

## Global rules — these apply to every class

**1. Label what is visible, not what you know.**
You know which walls have saturated brick behind them. The model sees pixels. If the
damp isn't visible in *this* frame, it isn't labelled in this frame.

**2. Overlapping boxes of different classes are expected and correct.**
Mould sits inside damp; blistering sits inside damp; all three occur in the same 30 cm
of wall. Label each independently, each box tight to its own evidence. Do not merge
them into one box, and do not pick "the dominant one".

**3. Tight boxes.**
The box touches the feature's pixels on all four sides. No margin. This is what
`mAP50-95` measures.

**4. Hard negatives get no boxes at all.**
The 46 frames with `negative_type` set in `areas.csv` are there to teach the model what
is *not* a defect. An empty image is a valid, valuable training example. If you find
yourself drawing a box on one of these, the definition is wrong — not the image.

**5. When genuinely unsure, don't label it, and write it down.**
An unlabelled ambiguous feature costs one missed instance. A wrongly labelled one
teaches a false rule across the whole dataset. Keep a running list; it becomes the
boundary-case section of v2.

**6. Partial features at the frame edge are labelled** if more than roughly a third is
visible, boxed to the visible extent only.

---

## 1. `crack`

| | |
|---|---|
| **Positive definition** | A linear fracture line in the paint or plaster surface, darker than the adjacent wall, with visible variation in width or direction along its length. |
| **Boundary cases — do NOT label** | • Shadow lines cast by shelves, brackets, door frames or hooks — uniform width, terminate where the object ends<br>• Construction joints: window/door architrave-to-wall junctions, plasterboard seams<br>• Straight lines from ink drop run-off<br>• Lines from overlapping paint layers<br>• **A line revealing an old crack painted over** — the surface is intact; this is a ghost, not a fracture |
| **Box rule** | One box per **continuous run**. A run that branches is one box if the branches stay within a reasonable bounding area; a branch that diverges into its own long path gets its own box. |
| **Minimum size** | **0.5 mm apparent width** at 40 cm (≈ 1 px at `imgsz=1280`). Below this the feature does not survive downscaling — see the resolution test. Length: at least 3 cm. |
| **Decision it feeds** | Where to open up, where to monitor for movement. |
| **Miss vs false alarm** | **Miss is worse.** A false alarm costs a closer look; a missed crack is a defect behind a wall someone is about to buy or sign for. Tune for recall. |

**The distinguishing test, for use at 11pm:** a crack *wanders* and varies in width. A
shadow is uniform and stops where the object stops. A construction joint is perfectly
straight and follows joinery.

**Expect mediocre `mAP50-95` on this class** — a diagonal crack fills maybe 15% of its
axis-aligned box. That is the box format, not your annotation.

---

## 2. `damp_patch`

| | |
|---|---|
| **Positive definition** | A region of wall surface darker or differently toned than the surrounding wall, with a soft, irregular, non-geometric boundary. |
| **Boundary cases — do NOT label** | • Transfer stains — sharp-edged, uniform tone, no boundary softness (your most common confuser, 7+ frames)<br>• Dirt residues and general soiling<br>• Shadows, including diffused ones — **verify by checking the same wall in another lighting condition**<br>• Patch repairs and skim work — tonal difference with no texture change<br>• Water or ink run-off streaks — linear and directional, not diffuse |
| **Box rule** | One box per **contiguous region**. Two patches separated by visibly clean wall are two boxes. A patch interrupted by furniture or a fitting is one box covering the visible extent. |
| **Minimum size** | **3 cm** across (≈ 70 px at `imgsz=1280`). |
| **Decision it feeds** | Where to take moisture readings; which zones need a full survey. |
| **Miss vs false alarm** | **Miss is worse.** Tune for recall. |

**DECIDE — the tide line.** Where a damp patch has a defined upper edge (rising damp),
does the box cover the whole affected region below the line, or just the visibly darker
band? I've assumed **the whole affected region**. Your near-floor holdout frames are
exactly this case, so the answer matters.

---

## 3. `mould`

| | |
|---|---|
| **Positive definition** | A cluster of dark speckling or spotting on the wall surface, distinct in texture from the underlying tone. |
| **Boundary cases — do NOT label** | • Accumulated dust on skirting boards and in corners<br>• Cobweb remnants and suspended dust<br>• Insect cocoons<br>• Dirt residues and transfer stains<br>• Small fixed objects — plastic beads, punch marks, nail heads |
| **Box rule** | One box per **colony or affected cluster**, not per spot. Individual spots are 1–3 mm and effectively invisible at `imgsz=1280`; the learnable feature is the speckled region. |
| **Minimum size** | **5 cm** cluster (≈ 115 px at `imgsz=1280`). Do not box isolated individual spots. |
| **Decision it feeds** | Habitability; ventilation assessment; urgency of the damp investigation. |
| **Miss vs false alarm** | **Miss is worse**, and more so than the other classes — this is the one with a health dimension. Recall first. |

**Note:** mould almost always sits inside a `damp_patch`. Both get boxes. This nesting
is correct and expected, and will show up in the confusion matrix as overlap — that is
not an error.

---

## 4. `blistering`

| | |
|---|---|
| **Positive definition** | Paint film lifted away from the substrate, forming raised areas with defined edges that cast small shadows in raking light. **The film is still attached** — nothing has come away. |
| **Boundary cases — do NOT label** | • Poor paint finish — orange peel, roller texture, patchy coverage. A defective *finish*, not a *failure*<br>• Paint texture change, including near door frames<br>• Lines from overlapping paint layers<br>• Surface irregularities in the plaster beneath intact paint<br>• **`peeling_paint`** — see below |
| **Box rule** | One box per **blistered patch**, not per blister. A single 10 mm blister is ~22 px and marginal; a patch of them is unmistakable. |
| **Minimum size** | **2 cm** patch (≈ 45 px at `imgsz=1280`). |
| **Decision it feeds** | Extent of surface preparation needed; corroborating evidence of moisture. |
| **Miss vs false alarm** | Roughly symmetric — this class feeds scope rather than safety. Slight recall bias for consistency with the rest. |

---

## 5. `peeling_paint`

| | |
|---|---|
| **Positive definition** | Paint film detached from the substrate, with **bare substrate visible** — the film has lifted away, broken open, or come off. |
| **Boundary cases — do NOT label** | • `blistering` — lifted but intact, nothing exposed<br>• Patch repairs where bare filler is visible but nothing has *detached*<br>• Plaster removed during the repair works — that's demolition, not paint failure<br>• Poor paint finish |
| **Box rule** | One box per **area of detachment**. |
| **Minimum size** | **2 cm** (≈ 45 px at `imgsz=1280`). |
| **Decision it feeds** | Extent of stripping and making good. |
| **Miss vs false alarm** | Symmetric. |

---

## The blistering / peeling_paint boundary — read this twice

These are the **same failure at two stages**, and they are the confusion this dataset is
most likely to produce. The test must be mechanical, because at image 140 your judgement
will not be:

> **Can you see bare substrate?**
> **No** → `blistering`. **Yes** → `peeling_paint`.

A blister that has burst and exposed plaster is `peeling_paint`. An area where both
occur side by side gets **two boxes**, each tight to its own evidence.

**DECIDE:** you have 124 `blistering` frames and 34 `peeling_paint` frames. If this
distinction proves unworkable in practice — if you find yourself guessing on more than
one image in ten — the honest move is to merge them into a single `paint_failure` class
and say so in the README. That would be a finding, not a retreat. Make that call inside
the first 25 images, not at 140.

---

## Do-not-label list (consolidated)

Drawn from the 25 confuser types documented in `areas.csv`. If you see one of these,
the correct action is **no box**:

| Confuser | Looks like | Why not |
|---|---|---|
| Transfer stains | damp | Sharp edge, uniform tone |
| Gradient shadow line | crack | Uniform width, moves with the light |
| Diffused shadows | damp | Moves with the light |
| Hook nail / shelf bracket shadow | crack | Terminates at the hardware |
| Door frame / architrave junction | crack | Follows joinery, perfectly straight |
| Window frame-to-wall joint | crack | Construction joint |
| Ink drop run-off | crack | Straight, directional |
| Overlapping paint layer line | crack | No fracture, surface intact |
| **Old crack painted over** | crack | Surface is intact — a ghost |
| Poor paint finish | blistering | Finish defect, not adhesion failure |
| Paint texture change | blistering | No lifting |
| Patch repairs | damp / peeling | Tonal difference, no texture change |
| Skirting board dust | mould | Dust, not growth |
| Cobwebs / suspended dust | mould | Not on the surface |
| Insect cocoon | mould | Discrete object |
| Punch mark | crack / mould | Mechanical damage, discrete |
| Plastic bead fixed to wall | mould | Attached object |
| Efflorescence | blistering | **Observed but excluded** — white crystalline deposit, visually distinct, too few instances to train. Do not box. |

---

## Annotation session discipline

- **Do the first 25 in one sitting**, then stop and re-read this document. Anything you
  contradicted, fix here first and re-do those 25.
- Work **batch by batch** (train, then val, then test). Don't interleave.
- Annotate hard negatives **deliberately**, not by skipping — open each one, confirm no
  box is warranted, move on. Skipping and "no box" look identical in the data but not in
  your confidence.
- Auto-labelling proposes; you adjudicate. Accepting an unreviewed box on a hard
  negative actively teaches the model that shadows are cracks.
- Keep a note of every case you found ambiguous. That list is your v2 boundary-case
  collection plan, and one of the three "next data improvements" the brief requires.

---

## Known limitations of these definitions

Stated here so the README can cite them rather than discover them:

1. **Near-floor regions are under-represented** in training. The lower ~40 cm appears
   mainly as hard negatives (skirting dust, patch repairs). Expect weak recall on
   near-floor damp — this is a coverage gap, not a threshold problem.
2. **Cracks below 0.5 mm apparent width are not detectable** at this capture distance
   and resolution. Established by measurement, not assumption.
3. **`blistering` / `peeling_paint` is a stage distinction**, and borderline cases exist
   where the film is lifted and cracked but not yet detached.
4. **Severity is not assessed.** No class carries a severity dimension. The model
   reports what is visible and where; a human assesses what it means.
