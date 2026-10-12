# Plant Disease Detection using a Transformer (Vision Transformer)

Detects plant diseases from leaf images and returns a full diagnosis (symptoms,
cause, treatment, prevention). This is the **transformer** half of a two-project
comparison: it is trained on **exactly the same images and the same stratified
train/val/test split** as the sibling `Plant-Disease-Detection-CNN` project — the
only difference between the two experiments is the architecture.

| | CNN project | This project |
|---|---|---|
| Architecture | custom CNN (542,287 params) | **Vision Transformer** (5,406,735 params) |
| Framework | TensorFlow / Keras | PyTorch |
| Input | 96×96×3 | 96×96×3 |
| Dataset | PlantVillage, 15 classes, 20,636 images | **identical (shared, read-only)** |
| Split | 16,508 / 2,064 / 2,064 stratified | **identical (shared, read-only)** |
| Test accuracy | 0.8488 (run-2 checkpoint) | **0.9477** |
| Weighted F1 | 0.8429 | **0.9477** |

## Shared data, no duplication

The dataset, the split CSVs and the diagnosis knowledge base are **not copied**
into this folder. They are read from the sibling CNN project at run time:

```
../Plant-Disease-Detection-CNN/data/PlantVillage   20,636 images, 15 classes
../Plant-Disease-Detection-CNN/data/split/*.csv    stratified 80/10/10
../Plant-Disease-Detection-CNN/data/diagnosis.json knowledge base
```

`src/preprocessing.py` locates them automatically (override with `--split-dir`),
then **verifies** before training that the splits are disjoint, exhaustive, that
every listed image exists and that `class_indices.json` matches the labels. It
records a SHA-256 fingerprint of each split's sorted image list in
`reports/data_provenance.json`:

| split | images | SHA-256 (first 16) |
|---|---|---|
| train | 16,508 | `132170992cae554b` |
| val | 2,064 | `0f3bc6da63c5df1d` |
| test | 2,064 | `abcc5a98a61e4874` |

Those fingerprints are what make the comparison honest: both projects provably
saw the same pixels in the same partitions. Preprocessing is identical too (RGB,
squash-resize to 96×96, scale to `[0, 1]`, no mean/std normalisation), so the
*only* variable is the network.

## Project structure

```
src/
  model.py             Vision Transformer (ViT) implementation
  preprocessing.py     split loading + verification, dataset, augmentation
  train.py             training pipeline (CLI)
  predict.py           inference module + CLI
  diagnosis.py         diagnosis knowledge-base lookup
app/
  main.py              FastAPI web app
  static/index.html    upload UI
models/                vit_plant_disease.pt, class_indices.json, config.json
reports/               history.json, classification_report.txt, confusion matrix,
                       data_provenance.json, MODEL_REPORT.md, train_console.log
notebooks/train.ipynb  interactive notebook mirroring train.py
```

## Setup

```
pip install -r requirements.txt
```

## Train

```
python src/train.py --epochs 32
```

That exact command produced the shipped checkpoint. Useful options:

```
python src/train.py --epochs 50 --arch vit_small     # bigger model (~37 min/epoch)
python src/train.py --patch-size 16                  # 37 tokens instead of 145 (faster)
python src/train.py --class-weighted --balanced-sampler
python src/train.py --resume models/last_checkpoint.pt
python src/train.py --time-budget-min 120            # stop cleanly after N minutes
```

`--resume` continues from `models/last_checkpoint.pt`, which is rewritten after
every epoch, so an interrupted run never loses more than one epoch.

## Predict from the command line

```
python src/predict.py --image path/to/leaf.jpg
```

Output: top class, confidence, top-3, per-class probabilities, a `confident`
flag (false below 0.5) and the full diagnosis.

## Run the web app

```
python -m uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 and upload a leaf photo. Corrupt or unsupported
uploads return HTTP 400 with a descriptive message; low-confidence predictions
are flagged in the UI.

## Architecture

```
input 96×96×3
  └─ patch embedding: Conv2d(3→192, kernel 8, stride 8)   → 12×12 = 144 patches
  └─ + learnable [CLS] token                             → 145 tokens
  └─ + learnable positional embeddings
  └─ 12 × pre-norm encoder block:
        LayerNorm → Multi-Head Self-Attention (3 heads × 64) → residual
        LayerNorm → MLP (192 → 768 → 192, GELU)           → residual
  └─ LayerNorm → Linear(192 → 15)
```

5,406,735 parameters. There is no convolution in the feature extractor — global
self-attention lets every patch attend to every other patch, so a lesion in one
corner can influence the representation of the whole leaf.

Patch size 8 (not the usual 16) because 96×96 is small: 8×8 patches preserve
12×12-pixel lesion detail that a 16×16 patch (36 tokens) would blur away. It
costs ~2.8× the compute per step and is worth it — patch 16 was benchmarked and
rejected.

### Training recipe

| | |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 (none on bias/norm/pos/cls) |
| Schedule | 3-epoch linear warmup → cosine decay to 1% of peak, per step |
| Loss | soft-target cross-entropy (Mixup/CutMix) + label smoothing 0.1 |
| Regularisation | stochastic depth 0.05, RandAugment(2, 9), RandomErasing 0.25, Mixup 0.2, CutMix 1.0 |
| Weight averaging | EMA (decay 0.9998), evaluated alongside raw weights each epoch |
| Gradient clipping | 1.0 |
| Batching | 64, bf16 autocast on AVX512-BF16 CPUs |

The model is trained **from scratch** — no ImageNet weights. That was deliberate:
pretrained ViT checkpoints expect 224×224 inputs, which is ~3.5× the compute per
step and was not feasible on this CPU-only machine, and a from-scratch model makes
the CNN-vs-transformer comparison about architecture rather than about who had a
bigger pretrained checkpoint.

## Results

32 epochs, 4h 26m on a 12-thread CPU (8.2 min/epoch), selected weights = epoch 32
(raw), validation accuracy 0.9486.

**Test accuracy 0.9477 · macro F1 0.9277 · weighted F1 0.9477** (2,064 images)

Best classes: `Tomato_healthy` F1 0.991, `Tomato__Tomato_YellowLeaf__Curl_Virus`
F1 0.988. Weakest: `Potato___healthy` F1 0.720 — it has only 15 test images
(4 after train/test carving of a 152-image class), so the estimate is noisy; the
CNN struggled on the same class for the same reason.

See `reports/MODEL_REPORT.md` for the full write-up and
`reports/confusion_matrix.png` for per-class errors.

## Classes

- Pepper: Bacterial spot, Healthy
- Potato: Early blight, Late blight, Healthy
- Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Mosaic virus (ToMV), Yellow leaf curl virus (TYLCV), Healthy
