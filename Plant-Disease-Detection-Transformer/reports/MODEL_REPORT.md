# Plant Disease Detection — Transformer Model Report

Generated for the Vision Transformer project (transformer half of the CNN-vs-transformer comparison).

## 1. Overview

A Vision Transformer (ViT) trained **from scratch** to classify PlantVillage leaf
images into 15 disease classes, plus a diagnosis layer and a FastAPI upload app.

The defining constraint of this project is **data parity**: it consumes the exact
same images, the same stratified 80/10/10 split and the same 96×96 preprocessing
as the sibling CNN project, so the architecture is the only independent variable.

Dataset: **20,636 leaf images**, 15 classes (3 Pepper, 3 Potato, 9 Tomato),
stratified → 16,508 train / 2,064 val / 2,064 test.

## 2. Data parity — how it is enforced

The dataset and split CSVs are **not duplicated**; they are read from
`../Plant-Disease-Detection-CNN/data/`. `src/preprocessing.py` refuses to train
unless it can prove:

1. no image appears in more than one split (splits are disjoint);
2. every path listed in the CSVs exists on disk;
3. `class_indices.json` matches exactly the labels found in the CSVs.

It then stores a SHA-256 fingerprint of each split's sorted image list
(`reports/data_provenance.json`):

| split | images | fingerprint (first 16 hex) |
|---|---|---|
| train | 16,508 | `132170992cae554b007c20705a571fb98cd370ef4d01e141c753dd4366b69721` |
| val | 2,064 | `0f3bc6da63c5df1d39966028a2452a5dfe8698faa5acef4b34bc97b4ebf99afb` |
| test | 2,064 | `abcc5a98a61e4874b586575440e025baac0d08a5cca6b28e2316978ad975ddd2` |

Preprocessing is also held identical: RGB convert, squash-resize to 96×96,
`/255` scaling, **no** mean/std normalisation, and the evaluation transform is
used at inference time too. A single scalar difference in preprocessing would
have made the comparison meaningless.

## 3. Architecture

ViT-Tiny sized for 96×96 inputs (`src/model.py`), **5,406,735 parameters**:

```
input 96×96×3
patch embedding  Conv2d(3→192, kernel=8, stride=8)     → 12×12 = 144 patches
                 + learnable [CLS] token                → 145 tokens
                 + learnable positional embeddings
12 × encoder block (pre-norm):
                 LayerNorm → MHSA (3 heads × 64 dim) → + residual
                 LayerNorm → MLP (192→768→192, GELU)  → + residual
head             LayerNorm → Linear(192→15)
```

**Why patch size 8.** The usual ViT uses 16×16 patches, which on a 96×96 image
gives only a 6×6 = 36-token grid — a 16-pixel patch is coarse relative to a
leaf spot. Patch 8 keeps a 12×12 grid (145 tokens with CLS). Benchmarked cost:
2.29 s/step versus 0.69 s/step for patch 16, so patch 16 was rejected as too
coarse for this resolution.

**Why from scratch, not ImageNet-pretrained.** Pretrained ViT checkpoints are
trained at 224×224 (196 tokens at dim 768). Matching that here would cost roughly
20× the compute per step — about 30–50 s/step on this CPU, i.e. days of training.
Training from scratch also keeps the comparison purely architectural instead of
partly a comparison of pretrained checkpoints.

**Why bf16.** Zen 4 (Ryzen 5 8640U) supports AVX512-BF16, which cut the step time
from 2.03 s to 1.48 s (~27%). Autocast is enabled automatically when the CPU
reports support, and can be forced with `--amp {auto,bf16,off}`.

## 4. Training recipe

| | |
|---|---|
| Optimizer | AdamW, lr 1e-3, betas (0.9, 0.999) |
| Weight decay | 0.05, **excluded** from bias, LayerNorm, `pos_embed`, `cls_token` |
| Schedule | 3-epoch linear warmup → cosine decay to 1% of peak, stepped per batch |
| Epochs / batch | 32 / 64 (257 steps per epoch) |
| Loss | soft-target cross-entropy over Mixup/CutMix targets, label smoothing 0.1 |
| Augmentation | RandomResizedCrop(96, scale 0.65–1.0) → h-flip → RandAugment(2 ops, mag 9) → RandomErasing(p 0.25) |
| Mixup / CutMix | α 0.2 / α 1.0, applied with p 0.5 |
| Regularisation | stochastic depth 0.05 (linearly ramped over depth) |
| Weight averaging | EMA, decay 0.9998 |
| Gradient clipping | global norm 1.0 |
| Precision | bfloat16 autocast |

Mixup/CutMix and EMA are the two choices that mattered most. A from-scratch ViT
has no convolutional locality prior, so it overfits a 16.5k-image dataset quickly;
mixing samples and averaging weights both act as strong regularisers.

**Model selection.** Both the raw and the EMA weights were evaluated on the
validation split each epoch, and whichever was better was kept (saved per epoch
in `models/vit_plant_disease.pt`). The winner was the raw weights at epoch 32.

## 5. Results

**Test accuracy 0.9477 · macro F1 0.9277 · weighted F1 0.9477** (2,064 images)

Validation accuracy 0.9486 at epoch 32 (selected). Total training time 4h 26m
(264.8 min), 8.2 min/epoch on 12 CPU threads.

Training curve (`reports/history.json`, plotted in the notebook):

| epoch | train loss | train acc | val loss | val acc | EMA val acc |
|---|---|---|---|---|---|
| 1 | 2.197 | 0.216 | 2.122 | 0.376 | 0.411 |
| 5 | 1.653 | 0.415 | 1.641 | 0.558 | 0.653 |
| 10 | 1.392 | 0.557 | 1.067 | 0.807 | 0.809 |
| 15 | 1.247 | 0.637 | 0.977 | 0.835 | 0.865 |
| 20 | 1.148 | 0.655 | 0.844 | 0.891 | 0.904 |
| 25 | 1.081 | 0.694 | 0.771 | 0.914 | 0.925 |
| 30 | 1.005 | 0.697 | 0.706 | 0.944 | 0.942 |
| 32 | 0.981 | 0.741 | 0.698 | **0.949** | 0.945 |

Train accuracy (0.741) sits well below validation accuracy (0.949) because the
training metric is measured on heavily augmented, mixup-blended batches while
validation is clean — the gap is an augmentation artefact, not overfitting. There
is no overfitting: validation loss was still falling when the cosine schedule
bottomed out at epoch 32.

### Per-class test results

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Pepper Bacterial spot | 0.989 | 0.920 | 0.953 | 100 |
| Pepper healthy | 0.979 | 0.953 | 0.966 | 148 |
| Potato Early blight | 1.000 | 0.870 | 0.931 | 100 |
| Potato Late blight | 0.914 | 0.950 | 0.931 | 100 |
| **Potato healthy** | 0.900 | 0.600 | **0.720** | **15** |
| Tomato Bacterial spot | 0.976 | 0.939 | 0.957 | 213 |
| Tomato Early blight | 0.938 | 0.910 | 0.924 | 100 |
| Tomato Late blight | 0.958 | 0.958 | 0.958 | 191 |
| Tomato Leaf Mold | 0.958 | 0.958 | 0.958 | 95 |
| Tomato Septoria leaf spot | 0.875 | 0.949 | 0.911 | 177 |
| Tomato Spider mites | 0.862 | 0.970 | 0.913 | 168 |
| Tomato Target spot | 0.947 | 0.886 | 0.915 | 140 |
| Tomato TYLCV | 0.985 | 0.991 | 0.988 | 321 |
| Tomato mosaic virus | 0.837 | 0.973 | 0.900 | 37 |
| Tomato healthy | 0.988 | 0.994 | 0.991 | 159 |

Strongest: `Tomato_healthy` (F1 0.991) and TYLCV (0.988). Weakest:
`Potato___healthy` (F1 0.720) — 15 test images, 9 of them misread as early/late
blight. That class has only 152 images in total, the smallest in the dataset, and
its recall of 0.600 is a data-volume problem rather than an architecture
problem; the CNN reported the same class as its weakest (F1 0.56).

Full table: `reports/classification_report.txt`. Errors: `reports/confusion_matrix.png`.

## 6. Comparison with the CNN project

Same data, same split, same preprocessing — different architecture.

| | CNN (custom) | ViT-Tiny (this project) |
|---|---|---|
| Parameters | 542,287 | 5,406,735 |
| Input | 96×96×3 | 96×96×3 |
| Test accuracy | 0.8488 (shipped checkpoint) | **0.9477** |
| Weighted F1 | 0.8429 | **0.9477** |
| Macro F1 | — | 0.9277 |
| Wall clock | ~2 min/epoch | ~8.2 min/epoch |

The transformer is ~10× larger and ~4× slower per epoch, and still ahead by
**9.9 accuracy points**. Reading the caveat honestly: the CNN's 0.8488 came from a
partially-failed run — its own report notes a run-1 configuration reached 0.9297
before being overwritten. Against that fairer CNN baseline the margin is +1.8
points, still a win but a narrow one, and obtained with 10× the parameters.

Where the two architectures genuinely differ is not the headline number but the
error profile. The CNN's weakest class was `Potato___healthy` (F1 0.56) and it
struggled to separate several Tomato viruses; the transformer reaches F1 0.90–0.99
on every Tomato class and its confusion matrix is nearly diagonal. Global
self-attention over 145 tokens lets it relate a small lesion to the whole leaf,
whereas a 5-layer CNN's receptive field is more local. The trade-off is data
efficiency: the transformer needs the heavy augmentation and EMA schedule above to
avoid overfitting, and it is materially more expensive per epoch.

## 7. What the model outputs

For an uploaded leaf image the system returns:

- `top_class` — e.g. `Tomato__Tomato_YellowLeaf__Curl_Virus`
- `top_confidence` — softmax probability of the top class
- `confident` — false when `< 0.5`, so non-leaf/ambiguous inputs are flagged
  rather than silently mislabelled (a blank white image returns
  `confident: false` at 44.3%)
- `top3` and `probabilities` — why the model picked what it picked
- `diagnosis` — crop, common name, healthy flag, symptoms, cause, treatment,
  prevention from `data/diagnosis.json`

Verified: 15/15 correct on randomly sampled test images; a corrupt upload and a
`.txt` upload both return HTTP 400.

## 8. AI/ML concepts used

| Concept | Where |
|---|---|
| Vision Transformer | `src/model.py` — patch embedding + 12 self-attention encoder blocks |
| Self-attention | every patch attends to every other patch (no locality prior) |
| Multi-head attention | 3 heads × 64 dims, fused SDPA kernel |
| Positional embeddings | learnable, added to the token sequence |
| CLS token | pooled representation for classification |
| Stochastic depth | `DropPath`, 0.05, ramped with depth |
| Layer normalisation | pre-norm blocks, stabilises deep training |
| Weight decay exclusions | no decay on bias/norm/positional/CLS parameters |
| AdamW | decoupled weight decay |
| Warmup + cosine schedule | per-step LR schedule, 3-epoch warmup |
| Mixup / CutMix | batch-level sample mixing with soft targets |
| Label smoothing | 0.1, applied to mixed soft targets |
| EMA weight averaging | `ModelEMA`, decay 0.9998 |
| RandAugment / RandomErasing | learned-policy-free augmentation |
| bfloat16 autocast | CPU AVX512-BF16 mixed precision |
| Gradient clipping | global norm 1.0 |
| Data provenance | SHA-256 split fingerprints, disjointness/existence assertions |
| Classification report | per-class precision/recall/F1 |
| Confusion matrix | per-class error analysis |
| Low-confidence rejection | 0.5 threshold surfaced in API and UI |

## 9. Reproduce

```
pip install -r requirements.txt
python src/train.py --epochs 32                       # retrain (~4.5 h on 12 CPU threads)
python src/predict.py --image path/to/leaf.jpg        # CLI
python -m uvicorn app.main:app --reload --port 8000   # web app
```

Training is resumable: `models/last_checkpoint.pt` is rewritten every epoch and
`python src/train.py --resume models/last_checkpoint.pt` continues an
interrupted run. Every run is seeded (`--seed 42`).
