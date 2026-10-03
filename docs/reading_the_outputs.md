# Reading the Outputs — a plain-language cheat sheet

How to read `results.png`, the confusion matrix, and the metrics tables, without assuming any computer-vision background. Written for this project, with its real numbers as the worked example.

---

## 1\. First, what a "loss" is

During training the model makes a guess, and a **loss** is a penalty score for how wrong the guess was. **Lower is better. Zero would be perfect.**

Three things to know:

- It is **not** a percentage and has no fixed scale. A box loss of 2.4 isn't "2.4 % wrong" and can't be compared to a classification loss of 3.4.  
- It is only meaningful **compared to itself** — the same loss, earlier or later in the same run, or between two runs of the same setup.  
- The model is trained to make loss go down. That's the whole of training: guess, measure the penalty, adjust slightly, repeat.

An **epoch** is one complete pass through all 134 training images. Run 2 did 112 of them.

---

## 2\. The three losses

A detector has to answer two different questions about every object, and the losses split along that line.

| Loss | The question it scores | In plain words |
| :---- | :---- | :---- |
| `box_loss` | **Where** is it? | Is the rectangle in roughly the right place and roughly the right size? |
| `dfl_loss` | **Where exactly** does it stop? | Is each of the four edges precisely placed? |
| `cls_loss` | **What** is it? | Having found something, is it called `crack` or `mould` or `damp_patch` correctly? (**D**istribution **F**ocal **L**oss) |

**Why there are two "where" losses.** YOLO doesn't predict an edge as a single number. For each edge it produces a spread of probabilities across possible positions — *"the left edge is probably here, possibly a bit left, unlikely further"* — and then takes a weighted average. `dfl_loss` (Distribution Focal Loss) scores how good that spread is.

So `box_loss` asks whether the rectangle is about right; `dfl_loss` asks whether the model is confident and correct about exactly where each side sits. Two aspects of the same thing, at different levels of fussiness.

---

## 3\. The two rows of `results.png`

Ten panels: **top row \= training, bottom row \= validation.**

- **Training** — the 134 images the model learns from. It sees these over and over and adjusts itself to fit them.  
- **Validation** — 44 images it is *checked* on after every epoch but **never learns from**. Think of it as a mock exam: it tells you how you're doing without teaching you the answers.

That distinction is what makes the plot diagnostic. Compare the two rows:

| What you see | What it means |
| :---- | :---- |
| Training loss falling, validation loss falling | Learning, and it transfers. Good. |
| Training loss falling, validation loss **rising** | **Overfitting.** It's memorising the 134 photographs instead of learning what a defect looks like. |
| Both flat or rising | Something is broken — bad learning rate, bad data. Not what happened here. |

**Ignore the first \~15 epochs.** Training starts from COCO weights with a brand-new detection head that knows nothing about your five classes, so early guesses are wild. That's the spike — `val/cls_loss` reaching 330 before collapsing to about 3\. It's warm-up, not signal.

---

## 4\. The four metrics (right-hand panels, and the CSV tables)

These are computed on validation only. Unlike losses, **higher is better**.

### First, the thing they all depend on: IoU

To score a prediction you need a **rule for "correct"**. **IoU** (**I**ntersection **o**ver **U**nion) measures how much a predicted box overlaps the true one:

> the area the two rectangles share, divided by the total area they cover between them

0 means no overlap. 1 means identical. 0.5 means they overlap about half.

The convention is that a prediction **counts as correct if IoU ≥ 0.50** — a reasonable but fairly forgiving bar.

### Precision and recall

|  | Question | Formula | Bad when |
| :---- | :---- | :---- | :---- |
| **Precision** | Of the boxes the model drew, how many were real? | correct ÷ drawn | it cries wolf |
| **Recall** | Of the real defects, how many did it find? | correct ÷ actually there | it misses things |

They trade against each other. Make the model bolder and recall rises while precision falls; make it cautious and the reverse. That's what \#6 of the error analysis measures.

Our run 2, at the default threshold: **precision 0.264, recall 0.105.** Of the 144 boxes it drew, 38 were real. Of the 363 real defects, it found 38\.

### mAP50 and mAP50-95

**AP** (**A**verage **P**recision) **summarises the whole precision-recall trade-off for one class into a single number**, instead of picking one operating point. **m** means averaged across all classes.

- **mAP50** — the lenient measure. It checks whether a predicted box overlaps the ground‑truth box by **at least 50 %** (IoU ≥ 0.50). That’s a **forgiving threshold**: if your box roughly covers the right region, it counts as correct. So mAP50 mostly tells you whether the model finds the right *place* for the defect.   
- **mAP50-95** — the strict measure. This is the **average mAP** computed at IoU thresholds from **0.50 to 0.95** in steps of 0.05. As the IoU threshold rises, the model must match the true box **more precisely** — edges, corners, and size all need to be accurate. High mAP50‑95 means the model not only finds the defect but also **pins down its boundaries** correctly. 

Our run 2: **mAP50 \= 0.086, mAP50-95 \= 0.033.**

**The ratio between them is informative.** 0.086 : 0.033 is about 2.6 : 1\. A model that places edges precisely has a ratio near 1.5 : 1; a wide gap says it finds roughly the right region but can't pin down the boundaries. That's the localisation problem in \#5.4, read straight off two numbers.

---

## 5\. The confusion matrix

Read the axis labels carefully — it's easy to get backwards.

- **y-axis (rows) \= what the model predicted**  
- **x-axis (columns) \= what was actually there**  
- values are normalised **down each column**, so each column sums to 1

Therefore:

| Cell | Meaning |
| :---- | :---- |
| diagonal | correct — predicted what was there |
| **`background` ROW** | **missed.** Something was there, the model predicted nothing. |
| **`background` COLUMN** | **false alarms.** Nothing was there, the model predicted something. |
| any other off-diagonal | **confused.** Found it, named it wrongly. |

Our run 2 background row: 0.88, 0.96, 0.78, 0.88, 0.86 — between 78 % and 96 % of real defects were missed.

**One trap.** Because each column is normalised, a class with few instances produces dramatic percentages from tiny counts. `peeling_paint → blistering` showed as 0.14, which sounds systematic; it was **one box** out of seven. Always check the absolute count before believing a normalised cell.

---

## 6. The phrase in §4.1, decoded

> *"Run 1 degrades three to four times faster on the localisation losses, and the two run 2
> measurements agree within about 13 %."*

In plain words:

Training makes the model better on the images it studies. **Overfitting** is when it keeps
getting better on those while getting *worse* on the images it is only tested on — it is
memorising the photographs instead of learning what a defect looks like.

You can measure how fast that goes wrong. Take the epoch where the validation loss is at its
lowest, then measure how much it climbs per epoch after that. A bigger number means the model
is going wrong faster.

For the two "where is it" losses, run 1 climbs three to four times faster than run 2. More
augmentation, slower decline. That's the evidence the augmentation change did what it was meant
to.

**Why "the two run 2 measurements" matters.** Run 2 was trained twice, identically, as a
reproducibility check. The two results differed — which tells you how much any single number in
this project can wobble just by being re-run. For these losses the wobble is about 13 %, and the
difference being claimed is 300 %. That gap is the reason the claim is believable.

The same check sank a different claim. For the "what is it" loss, the two identical runs gave
0.0023 and 0.0097 — further apart than the effect being measured. So nothing is claimed about
it. §4.6 has the full accounting.

---

## 7\. One-line answers, if asked

| Question | Answer |
| :---- | :---- |
| What's a loss? | A penalty score during training. Lower is better. Only comparable to itself. |
| What's an epoch? | One full pass through all the training images. |
| box vs dfl vs cls? | Where it is, where exactly its edges are, and what it's called. |
| Why two rows in `results.png`? | Top is what it learns from, bottom is what it's tested on but never learns from. |
| How do you spot overfitting? | Training loss falling while validation loss rises. |
| Precision vs recall? | Precision: how much of what it flags is real. Recall: how much of what's real it flags. |
| mAP50 vs mAP50-95? | Lenient vs strict overlap. The gap between them measures how precisely it places edges. |
| Why is mAP50-95 so much lower? | Because the boxes are approximately right rather than exactly right. |
| Confusion matrix orientation? | Rows are predictions, columns are truth. Background row \= misses. Background column \= false alarms. |

---

## 8\. What is *not* used here, and why

**Accuracy.** The obvious metric — "what percentage did it get right?" — is never used in object detection, and it's worth being able to say why.

Accuracy needs four counts: true positives, false positives, false negatives and **true negatives**. In detection there is no meaningful true negative. How many defects did the model correctly *not* find in a photograph of a blank wall? The question has no answer — you'd be counting every rectangle it could have drawn and didn't, which is effectively infinite.

So detection uses precision and recall, which need only the first three.