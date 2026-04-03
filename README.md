# Industrial Defect Detection — CNN MLOps Pipeline

Production-grade CNN MLOps pipeline for visual quality control on the MVTec AD dataset.

**Three models, one pipeline:**
- EfficientNet-B4 — defect classification (normal vs defective + defect type)
- PatchCore / WideResNet-50 — anomaly detection (train on normal-only)
- U-Net + EfficientNet encoder — pixel-level defect segmentation

**Full MLOps stack:**
DVC · Albumentations · SageMaker · MLflow · PTQ + QAT + ONNX · SageMaker Model Monitor

---

## Phase 1: Data Pipeline Setup

### 1. Clone & install

```bash
git clone https://github.com/your-username/defect-detection-mlops.git
cd defect-detection-mlops

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download MVTec AD

MVTec requires license acceptance — manual download:

```
https://www.mvtec.com/company/research/datasets/mvtec-ad
```

**Faster alternative (Kaggle):**
```bash
pip install kaggle
kaggle datasets download -d ipythonx/mvtec-ad
unzip mvtec-ad.zip -d data/raw/mvtec/
```

### 3. Initialize DVC

```bash
dvc init

# Configure S3 remote (replace with your bucket)
dvc remote add -d s3remote s3://your-bucket/defect-detection/data
dvc remote modify s3remote region us-east-1

# Push raw data to S3
dvc add data/raw/mvtec
dvc push
```

### 4. Run the data pipeline

```bash
# Full pipeline (all 4 stages in order)
dvc repro

# Or step by step:
python src/data/download.py    # Stage 1: verify dataset structure
python src/data/tile.py        # Stage 2: extract 512×512 patches
python src/data/split.py       # Stage 3: stratified image-level split
python src/data/validate.py    # Stage 4: data quality checks
```

### 5. Smoke test datasets

```bash
python src/data/dataset.py
```

Expected output:
```
[Classification] train samples: 18420
[Anomaly]        train samples (normal only): 12800
[Segmentation]   train samples (defect+mask): 5620
[OK] All datasets verified.
```

---

## Pipeline Outputs (Phase 1)

```
data/
├── raw/mvtec/                          ← original MVTec images (DVC tracked)
└── processed/
    ├── patches/                        ← 512×512 patch images + masks
    │   ├── bottle/train/good/
    │   ├── bottle/test/broken_large/
    │   └── bottle/ground_truth/
    ├── splits/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── metadata/
        ├── dataset_metadata.csv        ← full metadata (all splits)
        ├── tiling_stats.json           ← DVC metrics
        └── validation_report.json      ← DVC metrics
```

---

## Key Design Decisions

**Image-level splitting** — all patches from one source image go to the same split.
Prevents data leakage where the model memorizes image textures instead of learning defect patterns.

**512×512 patches with 64px overlap** — covers edge defects that would be cut
by non-overlapping tiling. Overlap patches are deduplicated at inference.

**WeightedRandomSampler** — MVTec is ~85% normal, ~15% defective.
Without rebalancing, the classifier learns to predict "normal" for everything
and gets 85% accuracy while being useless. Sampler forces balanced batches.

**Separate transforms per model** — PatchCore needs minimal augmentation
(preserve what "normal" looks like). Classifier needs aggressive augmentation
(small dataset, high overfitting risk). Segmentation needs joint image+mask transforms.

---

## Metrics (Phase 1 targets)

| Check | Target |
|---|---|
| All 15 categories present | ✓ |
| Validation report all pass | ✓ |
| Train/val/test ratio | 70/15/15 |
| Normal/defect ratio (train) | < 20:1 |
| Patch size | 512×512 |

---

## Next: Phase 2 — SageMaker Training

→ `pipelines/sagemaker_training.py`
