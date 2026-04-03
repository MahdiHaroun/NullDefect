### Data pipeline overview (`src/data`)

This folder contains everything needed to go from **raw MVTec AD images** to **PyTorch-ready datasets**:

- **`download.py`**: verifies that the raw MVTec dataset is downloaded and in the correct folder structure.
- **`tile.py`**: cuts big images into 512×512 patches (and mask patches).
- **`split.py`**: creates train/val/test splits at the *image* level and writes a single metadata CSV.
- **`transforms.py`**: Albumentations transforms for the three model types.
- **`dataset.py`**: PyTorch `Dataset` classes and a helper to build `DataLoader`s.

Below is a step‑by‑step explanation, written for someone who is new to MVTec and this codebase.

---

### 1. `download.py` — making sure the raw dataset exists

**Goal:** Check that the MVTec AD dataset is present on disk and looks like we expect.

Key pieces:

- **`load_config`**
  - Reads `configs/data.yaml`.
  - We use it to find:
    - `paths.raw_dir`: where the original MVTec dataset should live (e.g. `data/raw/mvtec`).
    - `dataset.categories`: list of category names (e.g. `bottle`, `cable`, …).

- **`verify_mvtec_structure(raw_dir, categories)`**
  - For each category in `categories`, it checks that these folders exist:
    - `raw_dir/<category>/train/good/` — normal train images.
    - `raw_dir/<category>/test/` — test images, with subfolders for each defect type.
    - `raw_dir/<category>/ground_truth/` — pixel‑level mask images for defects.
  - Prints warnings if something is missing.
  - Returns `True` if everything looks OK, otherwise `False`.

- **`count_images(raw_dir, categories)`**
  - Counts how many images exist per category:
    - Normal train images.
    - Normal test images (`test/good`).
    - Defect test images (all non‑`good` test subfolders).
  - Also lists the defect type names for each category.

- **Main script (`if __name__ == "__main__":`)**
  - Loads config.
  - If `raw_dir` is empty or missing:
    - Prints detailed manual download instructions (URL, where to extract, etc.).
    - Exits.
  - If the directory exists:
    - Calls `verify_mvtec_structure`.
    - If there are problems, shows instructions again and exits.
    - If all good, calls `count_images` and prints a nice summary table.
    - Tells you: **“Ready for next step: `python src/data/tile.py`”**.

So, `download.py` does **no downloading**, only **verification + instructions**.

---

### 2. `tile.py` — cut full images into 512×512 patches

**Goal:** Convert big MVTec images (up to 1024×1024) into many smaller 512×512 patches, with overlap. This:

- Increases the number of training examples.
- Keeps input sizes consistent.
- Makes segmentation easier because defects stay within a patch.

Important configuration values (from `configs/data.yaml`):

- `paths.raw_dir`: where the original MVTec images are.
- `paths.patches_dir`: where to save the generated patch images.
- `tiling.patch_size`: patch size (e.g. 512).
- `tiling.overlap`: overlap in pixels between patches.
- `tiling.min_defect_ratio`: minimum fraction of defect pixels required to *keep* a defect patch for segmentation.

Key functions:

- **`load_config`**
  - Same idea: read `configs/data.yaml` for paths and tiling params.

- **`extract_patches(image, patch_size, overlap)`**
  - Slides a window of size `patch_size × patch_size` over the image.
  - Step size (stride) = `patch_size - overlap`.
  - Returns a list of `(patch_image, row_start, col_start)`:
    - `row_start`, `col_start` are the top‑left pixel coordinates of the patch inside the original image.
    - These coordinates are later used to cut the corresponding mask patch at the **same** position.
  - Also adds patches aligned to the right/bottom edges so no pixels are lost.

- **`mask_has_enough_defect(mask, patch_size, min_ratio)`**
  - For a mask patch (`0` = normal, `255` = defect), counts how many pixels are > 0.
  - If `defect_pixels / (patch_size * patch_size) >= min_ratio`, returns `True`.
  - Used to ignore patches where the defect is almost invisible (too few defect pixels).

- **`tile_category(category, raw_dir, patches_dir, patch_size, overlap, min_defect_ratio)`**
  - Processes a single MVTec category (e.g. `bottle`).
  - Creates the output folders under `patches_dir/<category>/...`.
  - **Train split (normal only):**
    - Reads `raw_dir/<category>/train/good/*.png`.
    - Extracts patches with `extract_patches`.
    - Saves them as:
      - `patches_dir/<category>/train/good/<category>_train_good_<src_stem>_patch_<r>_<c>.png`
    - Updates stats (`train_patches`).
  - **Test split (normal + defects):**
    - Reads `raw_dir/<category>/test/<defect_type>/*.png`.
    - For each defect type:
      - If `defect_type == "good"` → **normal** test patches.
      - Else → **defect** test patches; there is a corresponding mask in `ground_truth/<defect_type>/`.
    - For **defect** images:
      - Load RGB image and grayscale mask.
      - Extract patches from the image as before.
      - Extract patches from the mask using `extract_patches(mask, ...)`.
      - For each image patch at position `(r, c)`:
        - Find the mask patch at the same `(r, c)`.
        - If the mask patch has **enough** defect pixels (via `mask_has_enough_defect`):
          - Save the mask patch under `patches_dir/<category>/ground_truth/<defect_type>/...`.
          - Save the corresponding image patch under `patches_dir/<category>/test/<defect_type>/...`.
        - Otherwise, **skip** that pair and increase `skipped_low_defect`.
    - Updates stats (`test_normal_patches`, `test_defect_patches`, `skipped_low_defect`).

- **Main script**
  - Parses optional `--category` (process one category) and `--config`.
  - Loads config and computes categories to process.
  - Loops over all categories, calling `tile_category` for each.
  - Aggregates stats, prints a per‑category summary and global totals.
  - Writes `data/processed/metadata/tiling_stats.json` with all stats.
  - Tells you: **“Next step: `python src/data/split.py`”**.

So, after running `tile.py`, you have a big directory of **image patches** and **mask patches** ready for splitting.

---

### 3. `split.py` — image‑level train/val/test split

**Goal:** Generate a **single metadata table** describing all patches, and split it into train/val/test **at the original image level**, not per patch.

Why image‑level splitting?

- If you split patches randomly, patches from the **same original image** could end up in both train and test.
- The model might then “cheat” by memorizing the image background instead of learning defect patterns.
- Splitting by `source_image` avoids that leakage.

Key concepts:

- Each row in the metadata represents **one patch** and includes:
  - `patch_path`: path to the patch image (relative to project root).
  - `mask_path`: path to the mask patch, or `None` if not applicable.
  - `category`: e.g. `bottle`.
  - `defect_type`: either `good` or the specific defect name.
  - `label`: `0` for normal, `1` for defective.
  - `label_name`: human‑readable label (e.g. `"normal"`, `"broken_large"`).
  - `source_image`: ID of the original full‑resolution image.
  - `original_split`: `"train"` or `"test"` from the raw MVTec dataset.

Main functions:

- **`load_config`**
  - Reads config for:
    - `paths.patches_dir`: where patched images live.
    - `paths.splits_dir`: where to save `train.csv`, `val.csv`, `test.csv`.
    - `paths.metadata_csv`: path for the combined metadata CSV.
    - `dataset.categories`: list of categories.
    - `splits.{train,val,test}`: split ratios.
    - `splits.seed`: random seed for reproducibility.

- **`build_patch_records(patches_dir, categories)`**
  - Walks the output of `tile.py` and builds the patch‑level records list.
  - For each category:
    - **Train / good:**
      - Reads all `.png` files from `patches_dir/<category>/train/good`.
      - Fills fields:
        - `patch_path`: full (relative) path to patch file.
        - `mask_path`: `None` (train normal patches have no mask).
        - `category`, `defect_type="good"`, `label=0`, `label_name="normal"`.
        - `source_image`: derived from patch filename (part before `"_patch_"`).
        - `original_split="train"`.
    - **Test / defects + good:**
      - Iterates over all subfolders under `patches_dir/<category>/test`.
      - For each `defect_type` folder:
        - `defect_type="good"` → normal test patches.
        - Else → defective patches.
      - For defective patches:
        - Tries to find matching mask patch:
          - `patches_dir/<category>/ground_truth/<defect_type>/<same_file_name>.png`.
        - If the mask exists, `mask_path` is set to that path; otherwise `None`.
      - Fills all metadata fields similarly.
  - Returns a big list of dictionaries; later turned into a Pandas DataFrame.

- **`stratified_image_split(df, train_ratio, val_ratio, test_ratio, seed)`**
  - **Input:** DataFrame with one row per patch, including `source_image`, `category`, `defect_type`, `label`.
  - Steps:
    1. Collapse to one row per `source_image` (unique original image).
    2. Create a **strata key**: `category + "__" + defect_type`.
    3. Merge very rare strata into `"rare__other"` to avoid errors during stratification.
    4. First split the unique images into:
       - `train` vs `val+test` using stratified `train_test_split`.
    5. Then split the `val+test` group again into separate `val` and `test`.
    6. Build a mapping: `source_image -> split_name`.
    7. Apply this mapping back to **all patches** from each image:
       - So every patch from `image_0001` now has the same `split` value.
  - Returns the original patch‑level DataFrame with an extra `split` column.

- **`main()`**
  - Loads config and paths.
  - Calls `build_patch_records`, builds `df`.
  - Prints how many total, normal, and defective patches exist.
  - Calls `stratified_image_split` to add the `split` column.
  - Drops any patches that could not be assigned (rare edge case).
  - Saves:
    - `train.csv`, `val.csv`, `test.csv` into `paths.splits_dir`.
    - Full combined `metadata_csv` to `paths.metadata_csv`.
  - Prints category breakdown and class imbalance by split.
  - Tells you: **“Next step: `python src/data/validate.py`”** (validation script).

After `split.py`, you have a **single CSV** containing every patch with:

- Which split it belongs to.
- Where its mask is (if any).
- All the labels and metadata needed by the datasets.

---

### 4. `transforms.py` — Albumentations transforms for each model

**Goal:** Provide three transform configurations, one for each model type:

- Classification (EfficientNet‑B4).
- Anomaly detection (PatchCore / PaDiM).
- Segmentation (U‑Net).

Shared pieces:

- **`IMAGENET_MEAN` / `IMAGENET_STD`**
  - Constant values used in `A.Normalize` so that inputs are compatible with ImageNet‑pretrained backbones.

Functions:

- **`get_classifier_transforms(image_size=512)`**
  - Returns a dict: `{"train": train_transform, "val": val_test_transform, "test": val_test_transform}`.
  - **Train transforms** are quite strong:
    - Resize to `image_size × image_size`.
    - Random flips, rotations, and `ShiftScaleRotate`.
    - Elastic transforms for realistic surface distortions.
    - Color jitter (brightness/contrast, hue/saturation/value).
    - Noise, blur, and `CoarseDropout` (similar to Cutout).
    - Normalize + convert to PyTorch tensor (`ToTensorV2`).
  - **Val/Test transforms**:
    - Only resize + normalize + tensor.
    - No heavy augmentation to keep evaluation stable.

- **`get_anomaly_transforms(image_size=512)`**
  - Also returns a dict with `train/val/test`.
  - **Minimal augmentations** because PatchCore needs clean, consistent normal images:
    - Train: resize, horizontal flip, normalize, tensor.
    - Val/Test: resize, normalize, tensor.
  - No color jitter, noise, or elastic transforms.

- **`get_segmentation_transforms(image_size=512)`**
  - Returns transforms that work on **both image and mask together**.
  - Uses `additional_targets={"mask": "mask"}` so Albumentations knows `mask` is a segmentation mask.
  - **Train:**
    - Resize.
    - Spatial transforms (flips, rotate, elastic, shift/scale/rotate) applied to both image and mask.
    - Color transforms (brightness/contrast, noise, blur) applied to **image only**.
    - Normalize + `ToTensorV2`.
  - **Val/Test:**
    - Resize, normalize, `ToTensorV2`, still with `additional_targets={"mask": "mask"}`.
  - Usage pattern:
    - `out = transform(image=img, mask=mask)`
    - `img_tensor = out["image"]`, `mask_tensor = out["mask"]`.

The `__main__` block in this file just does a quick sanity check with dummy arrays.

---

### 5. `dataset.py` — PyTorch datasets and DataLoaders

**Goal:** Use the metadata CSV from `split.py` to create PyTorch `Dataset` objects for:

- Classification model (EfficientNet‑B4).
- Anomaly detection model (PatchCore).
- Segmentation model (U‑Net).

All three datasets:

- Read the **same CSV** (`cfg["paths"]["metadata_csv"]`).
- Filter by `split` (`"train"`, `"val"`, `"test"`).
- Optionally filter by `category`.
- Return tensors ready for training.

Shared helper:

- **`load_config(config_path="configs/data.yaml")`**
  - Used in the smoke test at the bottom to get the CSV path and dataloader settings.

#### 5.1 `DefectClassificationDataset`

**Purpose:** Standard classification dataset: returns `(image_tensor, label)` for each patch.

- **Initialization (`__init__`)**:
  - Arguments:
    - `csv_path`: path to metadata CSV.
    - `split`: one of `"train"`, `"val"`, `"test"`.
    - `transform`: Albumentations transform (usually from `get_classifier_transforms()`).
    - `label_mode`: `"binary"` (default) or `"multiclass"`.
    - `category`: if set, filters to one MVTec category.
  - Steps:
    - Load CSV into a DataFrame.
    - Filter rows where `df["split"] == split`.
    - If `category` is given, filter `df["category"] == category`.
    - For `"multiclass"`:
      - Build `class_to_idx` and `idx_to_class` from unique `defect_type`s.
      - Set `num_classes`.
    - For `"binary"`:
      - `num_classes = 2`.

- **`__len__`**
  - Returns number of rows in `self.df`.

- **`__getitem__(idx)`**
  - Reads the row at index `idx`.
  - Loads the image from `row["patch_path"]` (OpenCV).
  - Converts BGR → RGB.
  - If `transform` is provided:
    - Passes `image` to Albumentations and extracts `["image"]`.
  - Else:
    - Converts numpy to PyTorch tensor and scales to `[0,1]`.
  - Builds the label:
    - If `"binary"`: use `row["label"]` (0 or 1).
    - Else: map `row["defect_type"]` through `class_to_idx`.
  - Returns `(img_tensor, label_tensor)`.

- **`get_class_weights()`**
  - Computes sample weights based on class frequency.
  - Inverse frequency weighting: rare classes get higher weight.
  - Used by `WeightedRandomSampler` to handle class imbalance.

#### 5.2 `AnomalyDataset`

**Purpose:** Dataset for anomaly detection (PatchCore style).

- **Initialization (`__init__`)**:
  - Arguments:
    - `csv_path`, `split`, `transform`, `category` (similar to above).
    - `normal_only`: if `True`, force only normal samples.
  - Logic:
    - Load CSV, filter by `split`.
    - Optionally filter by `category`.
    - If `normal_only` **or** `split == "train"`:
      - Keep only rows where `label == 0` (normal images).
    - Save resulting DataFrame.

- **`__getitem__(idx)`**
  - Loads the patch image (same as classification).
  - Applies transform if provided.
  - Returns a dict:
    - `"image"`: image tensor.
    - `"is_anomaly"`: 0.0 for normal, 1.0 for defect (from `label`).
    - `"category"`: category string.
    - `"defect_type"`: defect type string.

This matches the typical input format used by anomaly detection methods.

#### 5.3 `DefectSegmentationDataset`

**Purpose:** Dataset for segmentation model (U‑Net), returning `(image_tensor, mask_tensor)` for defect patches.

- **Initialization (`__init__`)**:
  - Arguments:
    - `csv_path`, `split`, `transform`, `category`.
    - `defect_only`: if `True` (default), only keep rows that:
      - Have `label == 1` (defective).
      - Have a non‑null `mask_path`.
  - Logic:
    - Load CSV, filter by `split`.
    - Optionally filter by `category`.
    - If `defect_only`:
      - `df = df[df["label"] == 1].dropna(subset=["mask_path"])`.
    - Save this filtered DataFrame.

- **`__getitem__(idx)`**
  - Loads image from `row["patch_path"]`, converts BGR → RGB.
  - Loads the mask from `row["mask_path"]`:
    - If missing or path doesn’t exist:
      - Creates an all‑zero mask (should not happen with `defect_only=True`).
    - Else:
      - Reads grayscale mask, then binarizes (`mask > 127` → 0/1).
  - Applies transform if provided:
    - Calls `self.transform(image=img, mask=mask)` (joint Albumentations).
    - Extracts `["image"]` and `["mask"]`.
  - Else:
    - Converts both image and mask to PyTorch tensors.
  - Converts mask to float (`0.0` or `1.0`), which is convenient for BCE‑style losses.
  - Returns `(img_tensor, mask_tensor)`.

#### 5.4 `build_classification_loaders`

**Purpose:** Convenience function to create train/val/test `DataLoader`s for classification.

- Arguments:
  - `csv_path`: metadata CSV.
  - `transforms`: dict from `get_classifier_transforms()`.
  - `cfg`: full config dict (for `batch_size`, `num_workers`, `pin_memory`).
  - `category`: optional category filter.
  - `use_weighted_sampler`: whether to use `WeightedRandomSampler` on the train split.

- Behavior:
  - For each split `"train"`, `"val"`, `"test"`:
    - Create `DefectClassificationDataset` with the corresponding transform.
    - For train:
      - If `use_weighted_sampler`:
        - Compute `weights = dataset.get_class_weights()`.
        - Create `WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)`.
        - Set `shuffle = False` (because sampler and shuffle cannot be used together).
    - Build a `DataLoader` with these options and store it in a dict.
  - Returns a dict: `{"train": loader, "val": loader, "test": loader}`.

#### 5.5 Smoke test (`if __name__ == "__main__":`)

When you run:

```bash
python -m src.data.dataset
```

The script:

1. Loads config and checks that `metadata_csv` exists.
2. Creates a train `DefectClassificationDataset`, prints:
   - Number of samples.
   - Shape of one image and the label.
3. Creates a train `AnomalyDataset`, prints:
   - Number of normal train samples.
   - Shape of one image and the `is_anomaly` flag.
4. Creates a train `DefectSegmentationDataset`, prints:
   - Number of defect+mask train samples.
   - If there are samples, prints shapes and unique mask values.
   - If there are **no** segmentation samples, prints a warning explaining why
     (no rows with `split == "train"`, `label == 1`, and a valid `mask_path`).
5. If all three parts run, it prints `[OK] All datasets verified.`.

---

### How everything fits together (high level)

1. **Verify dataset**: `python src/data/download.py`
   - Make sure raw images are placed correctly under `data/raw/mvtec`.
2. **Tile images**: `python src/data/tile.py`
   - Create 512×512 image patches and mask patches under `data/processed/patches`.
3. **Create splits + metadata**: `python src/data/split.py`
   - Build one big CSV with all patches and assign each to train/val/test.
4. **(Optional) Validate pipeline**: `python src/data/validate.py`
   - Run extra checks on the metadata (not shown here, but referenced).
5. **Smoke test datasets**: `python -m src.data.dataset`
   - Confirm that classification, anomaly, and segmentation datasets all load correctly.

After this, you are ready to plug these datasets and transforms into your training scripts.

