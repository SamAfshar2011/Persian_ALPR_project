# PlateHunter

> **A modular Iranian Automatic License Plate Recognition system for images and video — built with YOLO26, ConvNeXt V2, geometry-aware OCR, and multi-object tracking.**

PlateHunter does not treat license-plate recognition as one giant black box.
It breaks the job into three inspectable stages: **find the plate, find the characters, recognize the characters** — then adds format-aware decoding, confidence gating, and temporal tracking on top.

That makes the system easier to debug, easier to tune, and much harder to fool with one lucky prediction.

**Stack:** Python · PyTorch · Ultralytics YOLO · timm · OpenCV · SciPy · Apple Silicon / MPS

---

## What the pipeline does

```text
Full image / video frame
        │
        ▼
┌──────────────────────────────┐
│ Model 1 — Plate Detector     │
│ YOLO26s · 960 px             │
└──────────────┬───────────────┘
               │ plate crop
               ▼
┌──────────────────────────────┐
│ Model 2 — Character Detector │
│ YOLO26l · 608 px             │
└──────────────┬───────────────┘
               │ best 8 character boxes
               ▼
┌──────────────────────────────┐
│ Model 3 — Character Classifier│
│ ConvNeXt V2 Base · 32 classes│
└──────────────┬───────────────┘
               │
               ▼
     Position-aware decoding
     + confidence fusion
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   Image result   Video tracker
                  Kalman + Hungarian
                  + appearance ReID
                  + temporal OCR fusion
```

The important bit is that every failure has an address. If a plate is wrong, you can tell whether the problem came from **plate detection**, **character localization**, **character recognition**, or **tracking** instead of staring at one opaque OCR result and guessing.

---

# Architecture

## Stage 1 — Whole-plate detection

Model 1 detects one or more Iranian license plates in the original scene.

| Property | Value |
|---|---|
| Model | `yolo26s` |
| Task | Single-class plate detection |
| Training size | `960 × 960` |
| Optimizer | AdamW |
| Initial LR | `6e-4` |
| Batch size | `8` |
| Max epochs | `120` |
| Device target | Apple MPS |

The training notebook performs dataset validation before converting the original XML annotations into YOLO format. Bounding boxes are checked for malformed coordinates, tiny overflow, minimum size, and minimum plate area.

The detector uses deliberately conservative augmentation. License plates are not abstract art; aggressive flips, mosaic, or heavy perspective transforms can very quickly teach the model nonsense.

### Model 1 data snapshot

- **18,864** train image/XML pairs
- **4,175** validation image/XML pairs
- **1** initial pair issue in the supplied run
- **22,836** training plate instances in the generated YOLO label distribution

![Model 1 label distribution](Saved_models/model_1/train_run/labels.jpg)

---

## Stage 2 — Character localization

Model 2 receives a cropped plate and localizes the eight character regions.
It is intentionally a **single-class detector**: this stage answers *where is each character?*, not *what character is it?*

| Property | Value |
|---|---|
| Model | `yolo26l` |
| Task | Single-class character detection |
| Training size | `608 × 608` |
| Optimizer | AdamW |
| Initial LR | `3e-4` |
| Batch size | `12` |
| Max epochs | `250` |
| Box / class / DFL weights | `10.0 / 0.20 / 2.0` |

### Strict dataset validation

Before training, each sample is checked for:

- exactly **8 usable character boxes**
- malformed or out-of-bounds boxes
- duplicate / near-duplicate boxes
- extreme box-size outliers
- hidden or unsupported files
- previously blacklisted invalid samples

The supplied run discovered **30,318** train pairs and **5,559** validation pairs. After strict filtering, it kept **27,078** train samples and **4,985** validation samples, while recording **3,814** invalid samples/issues and **98** tiny bbox clamp events.

Ultralytics later rejected two extremely small `9 × 43` training images at dataloader time, leaving **27,076** usable training images. With eight characters per image, that produces **216,608 character instances**.

![Model 2 label distribution](Saved_models/model_2_fixed/train_run/labels.jpg)

### Exact-8 recovery

Inference does not simply take the eight highest-confidence boxes and hope for the best.

The character detector:

1. removes detections below a low confidence floor
2. applies NMS
3. uses row alignment and character-height consistency as geometry priors
4. keeps a small candidate pool
5. searches for the best eight-box subset
6. sorts the selected characters from left to right

If the stage still cannot produce **exactly eight** character boxes, the OCR result is rejected.

That sounds strict because it is. A confidently wrong plate is worse than an honest rejection.

---

## Stage 3 — Character classification

Each character crop is classified independently using **ConvNeXt V2 Base**.

| Property | Value |
|---|---|
| Architecture | `convnextv2_base` |
| Pretrained model | `timm/convnextv2_base.fcmae_ft_in22k_in1k` |
| Parameters | **87,725,600** |
| Classes | **32** |
| Train input | `224 × 224` |
| Eval / inference input | `288 × 288` |
| Optimizer | AdamW |
| Backbone LR | `8e-5` |
| Head LR | `4e-4` |
| Weight decay | `0.05` |
| Label smoothing | `0.03` |
| Gradient accumulation | `2` steps |
| EMA | enabled (`0.9997`) |

### Geometry-preserving preprocessing

Characters are **not stretched into a square**.

Each crop is resized while preserving its aspect ratio, centered on a square canvas, and padded using the median border color of the original crop. Training uses only mild character-safe affine and color augmentation — no horizontal flips, vertical flips, MixUp, or CutMix.

That matters because turning a glyph into a funhouse-mirror version of itself is a surprisingly efficient way to ruin an OCR classifier.

### Class imbalance handling

The training set is not perfectly balanced. Some classes have thousands of samples while a few rare classes have fewer than one hundred.

Instead of using full inverse-frequency weighting, the notebook combines:

- a moderated `WeightedRandomSampler`
- bounded class weights in the loss
- label smoothing
- EMA evaluation
- a one-epoch classifier-head warm-up before full fine-tuning

This gives rare classes extra attention without letting them completely hijack training.

### Data-integrity check

The classifier notebook also hashes train and validation files to detect **exact-byte leakage**.

In the supplied run:

- train images found: **89,467**
- validation images found: **2,937**
- exact train/validation duplicates detected: **130**
- final validation set after in-memory leakage removal: **2,807**
- invalid images: **0**

No leaked files are deleted or modified; they are simply excluded from validation in memory and written to a report.

![Model 3 training class distribution](Saved_models/model_3/plots/class_distribution.png)

![Model 3 confusion matrix](Saved_models/model_3/plots/confusion_matrix.png)

> **Validation note:** the supplied classifier artifacts show an extremely strong validation run, but one class (`S`) has zero validation support after leakage filtering and several rare classes have very small support. Treat validation metrics as evidence, not as a production guarantee. A frozen external test set is still the cleaner benchmark.

---

# Position-aware OCR decoding

Iranian plates in this project follow an eight-slot structure:

```text
DD L DDD DD
```

Using zero-based positions:

- positions `0, 1, 3, 4, 5, 6, 7` are digits
- position `2` is the letter / symbol slot

The classifier still produces probabilities for all 32 classes, but decoding only allows classes that are valid for the current position.

This is stronger than recognizing eight independent crops and validating the text afterward: impossible classes are blocked **during decoding**.

Persian and Arabic numerals are normalized to ASCII digits before the final formatted string is created.

---

# Confidence gating

A plate result has to survive all three stages.

For every candidate plate, PlateHunter computes:

- Model 1 plate confidence
- mean Model 2 character-detection confidence
- mean Model 3 character-classification confidence

The final OCR confidence is their **geometric mean**:

```text
final_confidence = geometric_mean(
    plate_confidence,
    mean_character_detection_confidence,
    mean_character_classification_confidence
)
```

Why geometric mean instead of a simple average?

Because one weak stage should not be able to hide behind two confident ones. If the plate detector is unsure, or the character detector is shaky, the final score feels it immediately. Bad predictions do not get free camouflage.

The default inference configuration also enforces minimum thresholds for each stage before accepting the OCR result.

---

# Batched inference

The upgraded inference path removes the old *one forward pass per character* bottleneck.

For a batch of candidate plates:

1. Model 2 runs across the plate crops
2. all character crops are flattened into one batch
3. Model 3 classifies them in batches of up to `32`
4. predictions are reassembled back into their original plates

That keeps the modular pipeline while avoiding eight tiny classifier forwards for every plate.

---

# Image inference

For a still image, PlateHunter:

1. detects every plate candidate
2. crops each plate with configurable padding
3. recovers the best eight character boxes
4. batches all character crops through ConvNeXt V2
5. applies position-aware decoding
6. computes stage and final confidence values
7. accepts or rejects each plate
8. draws the result on the image
9. saves crops and a CSV report

### Image outputs

```text
Saved_models/inference_outputs/
├── images/
│   └── <name>_annotated.png
├── crops/
│   └── <name>/
│       └── plate_XX/
│           ├── plate_crop.png
│           └── char_XX.png
└── reports/
    └── <name>_results.csv
```

The image report includes fields such as:

- acceptance / rejection status
- rejection reason
- normalized plate text
- formatted plate text
- Model 1 confidence
- mean Model 2 confidence
- mean Model 3 confidence
- final confidence
- plate bounding box

---

# Video tracking

Video mode adds a hybrid tracker on top of the same OCR stack.
The goal is not just to detect plates frame by frame — it is to keep the **same physical plate attached to the same `track_id`** through motion, short misses, and noisy detections.

## 8D Kalman motion model

Each active track uses a constant-velocity state:

```text
[cx, cy, w, h, vx, vy, vw, vh]
```

Measurements contain:

```text
[cx, cy, w, h]
```

The filter uses variable `dt`, Joseph-form covariance updates, and Mahalanobis gating.

## Hungarian assignment

Track-to-detection matching is solved with Hungarian assignment.
The matching score combines:

- IoU
- center-distance consistency
- size similarity
- plate appearance similarity
- detection confidence

Association runs in three rounds:

1. confirmed tracks ↔ high-confidence detections
2. tentative tracks ↔ remaining high-confidence detections
3. unmatched tracks ↔ lower-confidence detections

## Appearance descriptor and short-term ReID

Plate appearance is represented by a lightweight handcrafted descriptor built from:

- HOG / gradient structure
- intensity histogram
- coarse spatial intensity signature
- row projection
- column projection

If OpenCV HOG is unavailable, PlateHunter falls back to its own deterministic HOG-like gradient descriptor.

Confirmed tracks that disappear are kept in a short dormant memory. A new detection can reactivate an old ID when appearance and size similarity are strong enough.

No separate neural ReID model is required.

---

# Temporal OCR fusion

Video OCR is deliberately not rerun blindly on every frame.

Each track receives a quality score based on:

- detection confidence
- sharpness
- crop resolution
- plate aspect ratio

OCR cadence then adapts to track state:

- unread tracks are checked frequently
- tracks still building consensus are checked at a moderate interval
- stable tracks are checked less often
- a significantly better-quality frame can trigger an early OCR rerun

For temporal consensus, PlateHunter fuses the **full per-position probability matrices** from useful OCR observations instead of voting only on the final text string.

Very weak OCR frames are ignored so one terrible crop cannot poison an otherwise stable track.

---

# Video output path

Saved video mode produces:

```text
Saved_models/inference_outputs/
├── videos/
│   └── <name>_annotated.mp4
└── reports/
    └── <name>_track_summary.csv
```

The track summary contains:

- `track_id`
- confirmed state
- hit count
- track age
- OCR observation count
- normalized text
- formatted text
- consensus confidence
- best-frame confidence

On macOS, PlateHunter prefers **FFmpeg + `h264_videotoolbox`** when available, giving hardware-accelerated H.264 output. It also attempts to mux the original audio back into the annotated video.

If VideoToolbox is unavailable, it falls back to OpenCV video writing.

---

# Example outputs

### Model 1 — Plate detection

![Model 1 output](assets/model_1_output.png)

### Model 2 — Character localization

![Model 2 output](assets/model_2_output.png)

### Full pipeline

![Full pipeline output 1](assets/full_pipeline_output_1.png)
![Full pipeline output 2](assets/full_pipeline_output_2.png)
![Full pipeline output 3](assets/full_pipeline_output_3.png)
![Full pipeline output 4](assets/full_pipeline_output_4.png)
![Full pipeline output 5](assets/full_pipeline_output_5.png)
![Full pipeline output 6](assets/full_pipeline_output_6.png)
![Full pipeline output 7](assets/full_pipeline_output_7.png)
![Full pipeline output 8](assets/full_pipeline_output_8.png)
![Full pipeline output 9](assets/full_pipeline_output_9.png)
![Full pipeline output 10](assets/full_pipeline_output_10.png)

---

# Repository structure

```text
Persian_ALPR_project/
├── model_1.ipynb                 # plate detector training + validation
├── model_2.ipynb                 # character detector training + exact-8 evaluation
├── model_3.ipynb                 # ConvNeXt V2 classifier training + diagnostics
├── inference.ipynb               # image/video inference + tracking
├── assets/
│   └── ...
└── Saved_models/
    ├── model_1/
    │   ├── config.json
    │   ├── data.yaml
    │   ├── valid_samples.csv
    │   ├── invalid_samples.csv
    │   ├── bbox_clamp_report.csv
    │   ├── conversion_log.csv
    │   └── train_run/
    ├── model_2_fixed/
    │   ├── config.json
    │   ├── data.yaml
    │   ├── valid_samples.csv
    │   ├── invalid_samples.csv
    │   ├── bbox_clamp_report.csv
    │   ├── conversion_log.csv
    │   └── train_run/
    ├── model_3/
    │   ├── config.json
    │   ├── class_to_idx.json
    │   ├── class_to_idx_english.json
    │   ├── validation_metrics.json
    │   ├── per_class_accuracy.csv
    │   └── plots/
    └── inference_outputs/
        ├── images/
        ├── videos/
        ├── crops/
        └── reports/
```

Trained weights are intentionally not included in the public repository.

---

# Dataset layout

The training datasets are private and are therefore not bundled with the repository.

Expected high-level structure:

```text
Datasets/
├── DS_model_1/
│   ├── train/    # image + same-stem XML annotation
│   └── valid/
├── DS_model_2/
│   ├── train/    # cropped plate image + same-stem XML annotation
│   └── valid/
└── DS_model_3/
    ├── train/
    │   ├── 0/
    │   ├── 1/
    │   ├── ...
    │   └── ﻫ/
    └── valid/
        ├── 0/
        ├── 1/
        ├── ...
        └── ﻫ/
```

Model 3 uses a fixed explicit class order so filesystem ordering cannot silently change class indices.

---

# Installation

A clean virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
  numpy pandas pillow matplotlib tqdm \
  opencv-python scipy scikit-learn \
  torch torchvision timm ultralytics
```

Optional but recommended for fast video export on macOS:

```bash
brew install ffmpeg
```

The supplied notebooks were run on Apple Silicon with MPS. One captured environment used:

```text
Python      3.13.15
PyTorch     2.14.0
Torchvision 0.29.0
timm        1.0.29
Ultralytics 8.4.143
```

Other compatible versions may work, but these are the versions represented by the supplied notebook runs.

---

# Running the project

Launch Jupyter:

```bash
jupyter lab
```

Then use the notebooks as needed:

```text
model_1.ipynb   → train / inspect the plate detector
model_2.ipynb   → train / inspect the character detector
model_3.ipynb   → train / inspect the character classifier
inference.ipynb → run the complete image/video system
```

Before training Model 1 or Model 2 on another machine, update the configured local YOLO checkpoint directory in the notebook configuration. The supplied training configuration points to a machine-specific path.

Inference expects the trained weights at:

```text
Saved_models/model_1/train_run/weights/best.pt
Saved_models/model_2_fixed/train_run/weights/best.pt
Saved_models/model_3/best_model.pt
```

---

# Apple Silicon support

PlateHunter was built with macOS in mind rather than treating it as an afterthought.

The notebooks include separate device handling for:

- PyTorch / ConvNeXt
- Ultralytics YOLO
- CPU fallback where explicitly allowed
- MPS inference and training
- optional hardware video encoding through VideoToolbox

This makes the full pipeline practical on modern Apple Silicon machines without CUDA.

---

# Current limitations

PlateHunter is strong, but it is not magic. Current constraints include:

- the OCR logic assumes the standard **8-slot Iranian plate layout** used by the project
- Model 2 must recover exactly eight characters before OCR can continue
- severe blur, occlusion, glare, extreme perspective, or tiny plate crops can still break the pipeline
- position-aware decoding assumes the plate characters appear in the expected slot order
- handcrafted appearance ReID can struggle when viewpoint or lighting changes dramatically
- the current video code saves a **per-track summary**, not a full per-frame CSV log
- the current OpenCV overlay uses normalized display labels; dedicated Persian glyph rendering is not part of the upgraded inference path
- one classifier class (`S`) has zero validation samples after leakage filtering, and a few rare classes have very small validation support
- the private datasets and trained weights are not included, so the repository is not fully reproducible from a fresh clone by itself
- the supplied detector training CSVs are partial training snapshots, so they should not be presented as final detector benchmarks

Those limitations are documented on purpose. A system becomes more useful when it is clear about where it can fail.

---

# Training artifacts and diagnostics

The training notebooks generate structured artifacts instead of leaving everything buried inside notebook output.

Depending on the stage, these include:

- cleaned sample indexes
- invalid-sample reports
- bbox clamp reports
- YOLO conversion logs
- class-distribution CSVs and plots
- training logs
- validation metrics
- confusion matrix
- per-class accuracy
- classification report
- misclassification gallery
- best and last checkpoints

This makes it possible to audit the training pipeline instead of trusting one pretty accuracy number.

---

# Why PlateHunter is modular

A single end-to-end OCR network can be elegant, but a staged system has one major engineering advantage: **observability**.

If a result fails, PlateHunter lets you inspect:

```text
Was the plate detected?
        ↓
Were the eight characters localized correctly?
        ↓
Which character was misclassified?
        ↓
Did confidence gating reject it?
        ↓
Did the tracker preserve the identity over time?
```

For experimentation, debugging, dataset cleaning, and model replacement, that visibility is extremely valuable.

You can replace one stage without rebuilding the entire system around it.

---

# License

No standalone `LICENSE` file is currently included in the repository.
Until a license is added, do not assume the project grants open-source reuse rights beyond what is explicitly permitted by the repository owner.

---

# Author

Built by **Sam Afshar** as a practical computer-vision project for Iranian license plate recognition.

If you are here to inspect the code rather than just the screenshots: good choice. The interesting parts are the exact-8 geometry, position-aware decoding, leakage checks, and temporal probability fusion. 🚘
