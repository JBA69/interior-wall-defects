# Capture Protocol — Interior Wall Defects v1

How the photographs were taken, what the filenames encode, what the protocol intended, and
where the execution diverged from it.

**Project:** `interior-wall-defects` · MAICEN M4 U3
**Author:** José de Barros Aguiar
**Subject:** a storm-damaged residential interior, Lisbon, winter 2025–26

All counts in §2, §3, §5 and §6 were read directly off the published export, verified against
its SHA256, rather than taken from notes.

---

## 1. Equipment

| | |
|---|---|
| Camera | iPhone 12 Pro, 1× wide camera |
| Capture mode | still photo, no HDR bracketing, no portrait mode |
| Flash | none, in any frame |
| Stabilisation | none — handheld throughout |
| Export resolution | native capture, downscaled to 1280 px on the long edge at dataset export |
| Capture window | 14–20 September 2026, five sessions |

No flash means the lighting tokens describe the *only* illumination present in each frame, which
is what makes them analysable. A fill flash would have partially equalised `bright`, `dim` and
`elec`, and the lighting variable would have measured the flash rather than the room.

Handheld and unstabilised, at 20 cm from a surface, means a proportion of close frames carry
motion blur. That is the capture condition this dataset actually represents, and it is not
corrected for.

### 1.1 Capture sessions

| date | session | rooms |
|---|---|---|
| 14 Sep 2026 | defect capture | closet |
| 16 Sep 2026 | hard negatives — confusers | all six rooms |
| 17 Sep 2026 | hard negatives — sound wall | bedroom, living room, kitchen |
| 18 Sep 2026 | defect capture | hallway, kitchen, living room, office |
| 20 Sep 2026 | holdout frames | hallway, kitchen, living room |

**The closet was captured on its own day, four days before any other room's defects.** The
room-order confound described in §4 is visible here as a calendar fact, not an inference.

The damage dates from the winter 2025–26 storm, so every frame shows a defect six to nine months
after the event. These are **mature defects** — mould had time to establish, paint failure to
progress from blistering to detachment, damp to dry and leave a tideline. A dataset captured in
the days after a storm would show the same classes in earlier states and is not interchangeable
with this one.

### 1.2 Resolution

The 1280 px figure is not arbitrary. It was set by measurement before any training run:
a 0.5 mm hairline crack photographed at 40 cm survives downscaling to 1280 px as a feature
several pixels wide, and does not survive 640 px. The measurement is in
[`capture_resolution_test.md`](capture_resolution_test.md), with the test frame in
`crack_resolution_test.png`.

---

## 2. The capture grid

Each wall surface carrying a defect was photographed at a series of standoff distances crossed
with **three lighting conditions** — up to fifteen frames per surface.

### Distances

The room is a pronounced rectangle. The distance ladder is bounded by its geometry, not chosen
freely.

| token | standoff | frames in export | purpose |
|---|---|---|---|
| `c10` | 10 cm | 15 | paint-film detail at the limit of hand-held focus |
| `c20` | 20 cm | 76 | defect texture — speckling, crack edges |
| `c40` | 40 cm | 105 | the resolution-test reference distance |
| `c80` | 80 cm | **1** | a defect and its immediate surroundings |
| `c100` | 100 cm | 16 | a wall region, several defects in frame |
| `c300` | 300 cm | 9 | door wall to window wall — the room's long dimension, and therefore the **maximum face-on standoff the room physically allows** |
| `ctx` | corner diagonal, > 300 cm | **2** | as much of the long blind walls as the room permits |

**Two rungs barely exist.** `c80` appears in one frame and `ctx` in two. They were part of the
intended grid and were not executed at anything like the density of `c20` and `c40`, which
together carry 181 of the 224 frames. Within the training room the ladder that actually got
built is five rungs — 10, 20, 40, 100 and 300 cm — which is what §4 of
[`error_analysis.md`](error_analysis.md) describes.

**`ctx` is not another rung on the same ladder.** It is taken from a corner, so it varies two
things at once: it exceeds every available face-on standoff *and* it views the wall obliquely.
Every other framing is approximately perpendicular to the surface. This matters when reading
the distance findings in [`error_analysis.md`](error_analysis.md) §2.4 — frames in `ctx` are not
simply "far", they are far and foreshortened, with defects compressed along one axis and the
paint surface read at a glancing angle.

### Lighting

| token | condition |
|---|---|
| `bright` | daylight, shutters open |
| `dim` | reduced daylight — shutters half down, or end of day |
| `elec` | artificial light only — night, or shutters fully closed |

Lighting was partly opportunistic: `bright` and `dim` depend on the hour and the weather, and
sessions were scheduled around both. `elec` was available at any time.

---

## 3. Filename encoding

Every frame carries its capture conditions in its own name, so condition can be recovered from
the dataset alone without consulting the log:

```
CLOSET-D_c40_bright_clean_6997.jpg
│      │  │    │      │     │
│      │  │    │      │     └── capture index, continuous across the whole archive
│      │  │    │      └──────── constant marker, present on every frame
│      │  │    └─────────────── bright | dim | elec
│      │  └──────────────────── c10 | c20 | c40 | c80 | c100 | c300 | ctx
│      └─────────────────────── wall, A–G within the room
└────────────────────────────── room
```

This is what makes the capture-condition crosstabs in `02_Inference.ipynb` possible: the
analysis parses the tokens out of the filenames at evaluation time. Condition was recorded at
capture, not reconstructed afterwards.

**The wall is in the filename.** That matters more than it looks: the split-by-wall
recommendation in §8 is not a change to the capture protocol at all — the data already
identifies which wall every frame belongs to. Nothing needed collecting differently; the split
was simply drawn at the wrong level of an identifier the dataset already carried.

| room | walls | frames | frames per wall | split |
|---|---|---|---|---|
| CLOSET | 4 (A–D) | 134 | **33.5** | train |
| KITCHEN | 4 (A–D) | 34 | 8.5 | val |
| HALLWAY | 5 (A, B, C, E, G) | 10 | 2.0 | val |
| LIVING | 5 (A–E) | 31 | 6.2 | test |
| OFFICE | 4 (A–D) | 12 | 3.0 | test |
| BEDROOM | **1** (D) | **3** | 3.0 | test |

**33.5 frames per wall in training against 4.6 across the three test rooms.** That ratio is the
effective-sample-size argument of §7 expressed as a single number, and it is visible without
training anything. BEDROOM contributes one wall and three frames — present in the split table,
and absent from the dataset in any meaningful sense.

The per-frame log is `data/areas.csv` — 244 rows covering room, surface, distance, lighting,
defects present, and an `exclude` flag.

---

## 4. Room order, and the consequence

Six rooms were captured: **closet, kitchen, hallway, living room, office, bedroom.**

The closet came first, and was photographed most thoroughly, for a reason that seemed sound at
the time: it carried the most damage. It was also where the hard-negative taxonomy was built —
working at close range through one room's worth of defects is what surfaced the confusers worth
documenting.

Three things therefore coincide in that room: it was captured first, it was captured most
densely, and it held the highest concentration of defects. When the dataset was later split by
room to prevent near-duplicate leakage, the closet became the entire training set.

The result is that **room, capture purpose, and defect density are confounded with the split
boundary**, and no analysis of this dataset can separate them. That is the central finding of
[`error_analysis.md`](error_analysis.md) §2, and its cause is here, in the capture order.

---

## 5. Hard negatives — two kinds, captured separately

Background frames carry empty label files on purpose. They come from two distinct sessions with
two distinct jobs, and conflating them loses the design.

**Confuser negatives** (16 Sep, all six rooms) — surfaces that *resemble* a defect and are not
one: transfer stains from removed furniture, shadow lines along skirting, architrave and skirting
junctions, accumulated dust, tonal variation from uneven earlier painting, repair patches from
previous work. **18 types**, catalogued as they were found. These frames teach the decision
boundary: *this specific lookalike is not a defect.*

**Sound-wall negatives** (17 Sep, bedroom / living room / kitchen) — plain wall with nothing
notable on it at all. These teach the base rate: *most wall surface is not a defect.* A detector
shown only defects and lookalikes still has no evidence about ordinary wall.

Both kinds were captured in the same distance × lighting grid as everything else. The confuser
taxonomy was written down as it was built, so that a confuser identified in the closet was
recognised as the same confuser in the office.

### What the export actually contains

Counted directly from the published asset, SHA256 verified:

| split | images | background (empty label file) | with boxes |
|---|---|---|---|
| train | 134 | 20 | 114 |
| val | 44 | 3 | 41 |
| test | 46 | 23 | 23 |
| **total** | **224** | **46** | **178** |

**46 background frames, not 48.** The figure of "48 frames documenting 18 confuser types" is an
archive-side count and does not describe the published export; two of those frames did not reach
it.

**And the export does not distinguish the two kinds of negative.** Nothing in the filenames
separates a confuser frame from a sound-wall frame — both are simply frames with an empty label
file. The distinction is real and was deliberate at capture, but it lives in `areas.csv` and in
this document, not in the dataset. Anyone wanting to train on confusers specifically would have
to re-derive the split from the capture log. **Recording it as a filename token is the cheapest
fix for v2.**

---

## 6. Coverage as executed

The grid was the intent. The execution is uneven, and the gaps are consequential:

| gap | frames | consequence |
|---|---|---|
| **No `ctx` in training** | 0 train / 2 elsewhere | the model has never seen an oblique long-wall view |
| **No `c80` in training** | 0 train / 1 elsewhere | a gap in the middle of the ladder, not at its edge — and the rung barely exists at all |
| **`c300` trained, never evaluated** | 9 train / 0 elsewhere | the widest face-on framing is unmeasured |
| **`c10` concentrated in test** | 4 train / 10 test | the closest rung is denser in evaluation than in training |
| **44 of 436 evaluation instances (10 %) sit in framings absent from training** | 27 in `ctx`, 17 in the single `c80` frame | a tenth of the evaluation asks the model to extrapolate, not interpolate |
| Lighting mix differs between training and evaluation | — | and is confounded with room, so the effect cannot be isolated |

Measured consequence: recall falls monotonically with standoff — **0.132 at 20 cm to 0.000 at
100 cm**. The resolution test predicted this from the physics of the sensor; the evaluation
confirms it from the opposite direction.

Three logged frames never reached the export. They were identified by cross-checking the
published dataset against `areas.csv` in `01_Training.ipynb` §6b, and are marked `exclude=yes`
in the log rather than quietly dropped.

---

## 7. What the protocol got wrong

The distance × lighting grid was designed as a **robustness** protocol: show each defect under
many conditions so the model learns the defect rather than the photograph. Applied to a single
wall, that is exactly what it does.

The problem is structural, and it is worth writing out as a hierarchy:

```
grid (distances × lighting)   ⊂   wall surface   ⊂   room   ⊂   split boundary
```

Up to fifteen frames of one wall. Several walls per room. Six rooms. **The split was drawn at
the outermost level, two levels above the grid.**

Those two levels are the whole of the problem. The grid's job is to vary *how* a subject is
photographed. The split's job is to separate *which* subjects are trained on from those held
back. With two levels of aggregation between them, all fifteen views of a wall land on the same
side of the boundary — which is correct, and is what prevents near-duplicate leakage — but so
does every other wall in that room, and every defect on every one of those walls. The boundary
cannot distinguish "same wall photographed again" from "different wall in the same room," so it
treats both as the same subject.

That is why the closet's 1,430 training instances are many photographs of a much smaller number
of distinct physical defects. The effective sample size is the number of defects, and the room
boundary made every one of them train-only.

**The fix is to move the boundary down one level, to the wall surface.** The fifteen views of a
wall stay together, so leakage is still impossible; but walls distribute across splits, so no
single room becomes the entire training domain. The dataset already carries the wall identifier
needed to do it (§3).

**And walls are not merely a finer unit — they are a materially different one.** An exterior wall
beneath a window fails by water ingress at the frame. An interior partition fails by
condensation in a cold corner, or by movement at a junction. The window wall and the door wall
of a rectangular room are different damage mechanisms wearing the same paint. Splitting by room
discards that variation by putting whole mechanisms on one side of the boundary; splitting by
wall preserves it, because each unit carries its own mechanism. A model trained across wall
types sees the range of causes; a model trained on one room's walls sees whichever causes that
room happens to have.

So the error is in neither the grid nor the effort. It is in capturing **wall by wall to
completion within one room at a time**, when the unit that needed to be distributed was the wall
itself.

---

## 8. Protocol for v2

Ordered by what the measurements here justify:

1. **Split at the wall, not the room**, and cross the grid with location. Every room at every
   distance and lighting condition, with the wall as the unit assigned to a split. This is the
   single change that matters, and it is a change in capture *order* and *bookkeeping*, not in
   capture effort.
2. **Add `ctx` and `c80` to every room**, so no evaluation framing is absent from training.
3. **Evaluate `c300`**, or drop it. A framing that is trained and never measured earns nothing.
4. **Count distinct physical defects, not boxes**, and record a defect identifier in
   `areas.csv` so that the same crack photographed fifteen times is traceable as one subject.
   Effective sample size then becomes a number the dataset reports about itself.
5. **Photograph wall by wall across rooms rather than room by room to completion.** A wall is a
   finer unit than a room, still prevents near-duplicate leakage, and makes a balanced split
   achievable without sacrificing a whole room to one side of it.
6. **Log lighting as measured, not categorised.** Three ordinal buckets cannot be regressed
   against. An illuminance reading, even from a phone sensor, can.
7. **Mark the negative's kind in the filename** — `_conf_` against `_sound_` — so the published
   export carries a distinction that currently survives only in the capture log.
8. **Either execute `c80` and `ctx` properly or drop them from the grid.** One frame and two
   frames are not rungs; they are gaps that look like coverage in a protocol description.

Items 1 and 2 are prerequisites for anything in [`error_analysis.md`](error_analysis.md) §7
having a dataset to act on.

---

*The protocol above is reported as executed, including where it failed. The grid was carried out
as designed; the design was crossed with the wrong variable, and that is visible only in
hindsight and only because the splits were drawn to expose it.*
