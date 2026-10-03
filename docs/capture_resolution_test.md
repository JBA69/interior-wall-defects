# Capture Resolution Test — Hairline Cracks

**Question:** does a hairline crack survive downscaling to `imgsz=640`, or does the dataset need 1280?

**Answer: 1280.** At 640 the weakest sections of the crack fall to 1–2 grey levels, which is not learnable. At 1280 the same sections hold 3–4 levels and the strong sections 12–35.

---

## 1. Why this test exists

YOLO resizes every image to a fixed `imgsz` before the network sees it. Detail finer than the output pixel grid is destroyed at that step — no amount of labelling, training or threshold tuning recovers it. For region-scale defects (damp patches, mould colonies, blistered areas) this is irrelevant; they are tens of centimetres across. For hairline cracks it is the decision that determines whether the class is viable at all.

The instructor's guidance was that 640 is the default and hairline cracks may need 1280. This test checks that against the actual capture setup rather than accepting it.

---

## 2. Setup

| | |
|---|---|
| Camera | iPhone 12 Pro, main camera (4.2 mm, 26 mm equiv., f/1.6) |
| Capture | 3024 × 4032, 5.3 MB JPEG, ISO 320, 1/60 s, indoor |
| Subject | Fine crack on painted plaster, below ceiling cornice |
| Scale reference | Steel tape in frame, ≈ 64 px/cm at full resolution |
| Distance | ≈ 45 cm, frame covering ≈ 48 cm across the short side |

**Method.** Sample six positions along the crack. At each, take the darkest pixel in a narrow band crossing the crack, and the median of that band as the local wall value. Contrast is the difference in grey levels (0–255). Repeat on the same image downscaled to 1280 and 640 on the long side, using identical fractional coordinates. Noise σ estimated from adjacent-pixel differences on a clean wall patch.

Resolution matters for validity: the test must run on the **full-resolution original**, not a phone-share export. An earlier run on a 1536 × 2048 compressed copy gave systematically lower contrast and would have misled in both directions.

---

## 3. Results

Contrast in grey levels (0–255). Higher is better.

| Position | source (3024×4032) | imgsz=1280 | imgsz=640 |
|---|---|---|---|
| y = 0.18 | 11 | 12 | 7 |
| y = 0.24 | 13 | 6.5 | 3.5 |
| y = 0.29 | 8 | 4 | **2** |
| y = 0.35 | 9 | 3 | **1** |
| y = 0.46 | 9 | 12 | 7 |
| *y = 0.40* | *55* | *35* | *11* |

Noise σ: 3.03 (source), 1.46 (1280), 0.67 (640).

**y = 0.40 is not the crack.** The column scan caught adjacent paint-loss flecking. It is reported here because it is useful: a blister/spalling-type defect survives 640 at 11 levels without difficulty. Only the hairline is at risk.

**Visual check** (source / 1280 / 640, same crop, same display size): the crack is clear in the source, soft but continuous and traceable at 1280, and at 640 reduced to a row of barely-distinguishable pixels — visible to a human completing the line, not a rendered feature.

---

## 4. Reading the numbers

**Use contrast, not SNR.** An SNR column (contrast ÷ noise σ) was computed and is misleading here: downscaling averages sensor noise away at roughly the same rate it averages the crack away, so the ratio stays flat by construction and appears to show 640 performing as well as 1280. It does not. The absolute signal is what the network has to work with.

**Why 1–2 levels is not learnable.** One grey level is a single 8-bit quantisation step. JPEG's 8×8 block quantisation alone can erase a feature at that amplitude, and any brightness or exposure augmentation certainly will. The feature has to survive the full preprocessing chain, not just the resize.

**The crack is not uniform.** Contrast varies 8–13 levels along its length at source, and 3–12 at 1280. This is a property of the defect — the upper extent is genuinely fainter — not a measurement problem. Expect correspondingly uneven recall.

**Elevated source noise (3.03).** Higher than the 1280 figure, which averaging alone cannot produce. This is JPEG artifacting at full resolution, consistent with ISO 320 indoor capture. It does not affect the conclusion.

---

## 5. Decisions

**Train and infer at `imgsz=1280`.**

- Roughly 4× the compute of 640. On a Colab T4, reduce `batch` to 4–8 if memory limits are hit.
- Inference **must** match training resolution. A mismatch degrades results silently.
- **Roboflow resize preprocessing must be set to 1280**, not the 640 given in the course homework. Exporting at 640 and training at 1280 upscales a 640 image back up and recovers nothing — strictly worse than either option alone.
- Keep brightness/exposure augmentation ranges tight (≈ ±10%). At 3–4 levels, the weakest crack sections have little margin.

**Minimum size rule for `class_definitions.md`:** at ≈ 40 cm capture distance with the iPhone 12 Pro main camera, cracks are reliably learnable down to roughly **0.5 mm apparent width at `imgsz=1280`**, and not below. Cracks fainter than this may be annotated but should be expected to have lower recall.

**Expect uneven recall on `crack`** and state it in the README. It follows from the contrast variation along a single crack, not from model quality.

---

## 6. Trade-off deliberately rejected

1280 is applied to the **whole dataset**, though only one class requires it. Damp, mould and blistering were all comfortable at 640 and would have trained four times faster.

The alternative — a 640 model for region-scale defects plus a separate 1280 pass for cracks — would be faster in deployment and is the better engineering answer at scale. It was rejected here because a single model with one defensible resolution is simpler to document, defend and hand over, and because training time is not the binding constraint on this project.

---

## 7. Method notes for repeating this

**Honour EXIF orientation.** `Orientation: 6` means the file is stored landscape with a 90° flag. Loading raw pixel data without applying the flag gives a sideways image and, worse, a wrong long-side downscale factor. Use `ImageOps.exif_transpose()`. This is the same failure that Roboflow's **Auto-Orient** preprocessing exists to prevent — leave it enabled when generating dataset versions, since an annotation tool and a training pipeline disagreeing about rotation puts every box 90° out.

**Verify the file is the original.** Check `im.size` reads 3024 × 4032 and that `LensModel` is populated. Phone-share and cloud-sync exports routinely return recompressed, downscaled copies with EXIF stripped. Confirmed here by the file growing from ~500 KB to ~5.3 MB on re-import.

**Check the scan found the right feature.** The automatic search takes the darkest column in a search band. It will happily return a tape edge, a wall corner, a shadow boundary or an adjacent defect — as it did at y = 0.40. Always display the located strip before trusting the table.

**Do both tests.** The numeric measurement and the visual check at 100% are independent, and agreement between them is what makes the conclusion trustworthy. Here both said the same thing.

---

## 8. Reproduction

Notebook: `notebooks/capture_resolution_test.ipynb`
Figure: `docs/crack_resolution_test.png` (source / 1280 / 640, identical crop)
Test image: `CLOSET-xx` — full-resolution original, unmodified, in `00_raw/`

---

## Postscript — confirmed in evaluation

This test predicted a detection floor from pixels per centimetre at 40 cm. Inference on the
trained model found recall declining monotonically with capture distance — 0.132 at 20 cm,
0.085 at 40 cm, 0.059 at 80 cm, 0.000 at 100 cm — which is what that floor implies once a
defect subtends fewer pixels than the threshold established here.

See `docs/error_analysis.md` §2.4.

---