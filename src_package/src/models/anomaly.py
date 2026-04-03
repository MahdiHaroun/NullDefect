"""
src/models/anomaly.py

PatchCore anomaly detection — wrapper around anomalib.

What PatchCore does (no gradients, no epochs):
  1. Pass all normal training images through WideResNet-50
  2. Extract feature patches from layer2 + layer3
  3. Store a coreset (random subset) of those feature vectors in a memory bank
  4. At inference: compare new image features to memory bank via k-NN
  5. Anomaly score = distance to nearest neighbors
     High distance → far from "normal" → anomaly detected

Why anomalib?
  Implementing PatchCore from scratch requires:
    - Greedy coreset subsampling (O(n²) memory complexity)
    - Efficient approximate nearest neighbor search
    - Per-category threshold calibration
  anomalib handles all of this. We configure it via YAML and call it.

API note (anomalib >= 1.0):
  - Class renamed: MVTec → MVTecAD
  - image_size, center_crop, normalization removed from datamodule
    These are now handled by the model's built-in pre-processor.
"""

import json
import os
from pathlib import Path


class PatchCoreTrainer:
    """
    Wrapper around anomalib's PatchCore for integration into our pipeline.

    Usage:
        trainer = PatchCoreTrainer(cfg, category="bottle")
        results = trainer.train_and_evaluate()

        # All 15 categories:
        summary = trainer.train_all_categories(categories)
    """

    def __init__(self, cfg: dict, category: str):
        self.cfg       = cfg
        self.category  = category
        self.model_cfg = cfg["model"]
        self.data_cfg  = cfg["data"]

        # Verify anomalib is installed at construction time
        # Actual imports are deferred to build_* methods to avoid
        # old MVTec reference issues across anomalib versions.
        try:
            import anomalib  # noqa: F401
        except ImportError:
            raise ImportError(
                "anomalib is not installed. Run: uv add anomalib"
            )

    def _artifact_root(self) -> Path:
        """
        Save under SM_MODEL_DIR on SageMaker so artifacts are uploaded to S3.
        Falls back to local checkpoints directory for local runs.
        """
        sm_model_dir = os.environ.get("SM_MODEL_DIR")
        if sm_model_dir:
            return Path(sm_model_dir) / "anomaly"
        return Path("checkpoints/anomaly")

    def build_datamodule(self):
        """
        Build the MVTecAD datamodule.

        anomalib >= 1.0 change:
          - Class renamed from MVTec to MVTecAD
          - image_size / center_crop / normalization removed —
            the model's pre-processor handles resizing and normalization
        """
        from anomalib.data import MVTecAD

        return MVTecAD(
            root=self.cfg.get("paths", {}).get("raw_dir", "data/raw/mvtec"),
            category=self.category,
            train_batch_size=self.cfg["training"]["batch_size"],
            eval_batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["num_workers"],
        )

    def build_model(self):
        """
        Build PatchCore model.

        The model's internal pre-processor handles:
          - Resize to 256 × 256 (default)
          - Center crop to 224 × 224
          - ImageNet normalization
        All of this happens automatically inside anomalib.
        """
        from anomalib.models import Patchcore

        return Patchcore(
            backbone=self.model_cfg["backbone"],
            layers=self.model_cfg["layers"],
            coreset_sampling_ratio=self.model_cfg["coreset_sampling_ratio"],
            num_neighbors=self.model_cfg["num_neighbors"],
        )

    def train_and_evaluate(self) -> dict:
        """
        Run PatchCore for one MVTec category.

        "Training" = one forward pass to build the memory bank.
        No gradients, no epochs, no loss function.
        Takes minutes not hours.

        Returns dict with image_auroc and pixel_auroc.
        """
        from anomalib.engine import Engine

        datamodule = self.build_datamodule()
        model      = self.build_model()

        artifact_root = self._artifact_root()
        engine = Engine(
            default_root_dir=str(artifact_root / self.category),
        )

        # Build memory bank (single forward pass through all normal train images)
        print(f"\n[PatchCore] Building memory bank for '{self.category}'...")
        engine.fit(model=model, datamodule=datamodule)

        # Evaluate on test set (normal + all defect types)
        print(f"[PatchCore] Evaluating '{self.category}'...")
        test_results = engine.test(model=model, datamodule=datamodule)

        # Extract metrics from anomalib results dict
        results = {}
        if test_results:
            r = test_results[0] if isinstance(test_results, list) else test_results
            # anomalib 1.x metric keys
            results["image_auroc"] = float(r.get("image_AUROC", r.get("AUROC", 0.0)))
            results["pixel_auroc"] = float(r.get("pixel_AUROC", r.get("pixel_AUROC", 0.0)))

        print(
            f"[PatchCore] {self.category}: "
            f"image_AUROC={results.get('image_auroc', 0):.4f}  "
            f"pixel_AUROC={results.get('pixel_auroc', 0):.4f}"
        )

        return results

    def train_all_categories(self, categories: list[str]) -> dict:
        """
        Train and evaluate PatchCore for all 15 MVTec categories.
        Each category gets its own memory bank (stored in checkpoints/anomaly/<category>/).

        Returns summary dict with per_category results and mean metrics.
        """
        all_results = {}

        for category in categories:
            self.category = category
            try:
                results = self.train_and_evaluate()
                all_results[category] = results
            except Exception as e:
                print(f"[PatchCore] ERROR on '{category}': {e}")
                all_results[category] = {"image_auroc": 0.0, "pixel_auroc": 0.0, "error": str(e)}

        # Compute mean metrics (exclude failed categories)
        image_aurocs = [
            r["image_auroc"] for r in all_results.values()
            if "image_auroc" in r and r["image_auroc"] > 0
        ]
        pixel_aurocs = [
            r["pixel_auroc"] for r in all_results.values()
            if "pixel_auroc" in r and r["pixel_auroc"] > 0
        ]

        summary = {
            "per_category":     all_results,
            "mean_image_auroc": sum(image_aurocs) / len(image_aurocs) if image_aurocs else 0.0,
            "mean_pixel_auroc": sum(pixel_aurocs) / len(pixel_aurocs) if pixel_aurocs else 0.0,
        }

        # Save results JSON (used by Phase 3 evaluation)
        artifact_root = self._artifact_root()
        out_path = artifact_root / "results_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*50}")
        print("  PatchCore — All Categories Summary")
        print(f"{'='*50}")
        print(f"  Mean Image AUROC : {summary['mean_image_auroc']:.4f}")
        print(f"  Mean Pixel AUROC : {summary['mean_pixel_auroc']:.4f}")
        print(f"  Results saved to : {out_path}")

        return summary


