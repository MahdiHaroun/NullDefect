"""
pipelines/sagemaker_training.py  (SageMaker SDK V3 — corrected)

V3 pattern: ALL config goes into ModelTrainer constructor.
trainer.train() only accepts input_data_config + wait.

Usage:
    uv run python pipelines/sagemaker_training.py --model classifier
    uv run python pipelines/sagemaker_training.py --model classifier --dry-run
"""

import argparse
import os
import subprocess
import sys
import uuid

import yaml

# ── Config ────────────────────────────────────────────────────────────────────

S3_BUCKET  = os.environ.get("SAGEMAKER_S3_BUCKET", "your-bucket-name")
S3_PREFIX  = "visionforge"
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

SAGEMAKER_ROLE = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerFullAccess",
)



# AWS DLC PyTorch 2.1 GPU image — eu-central-1
PYTORCH_TRAINING_IMAGE = (
        "763104351884.dkr.ecr.eu-central-1.amazonaws.com/"
    "pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
)

MODEL_CONFIG = {
    "classifier": {
        "entry_script": "src/training/train_classifier.py",
        "config_file":  "configs/train_classifier.yaml",
    },
    "anomaly": {
        "entry_script": "src/training/train_anomaly.py",
        "config_file":  "configs/train_anomaly.yaml",
    },
    "segmentation": {
        "entry_script": "src/training/train_segmentation.py",
        "config_file":  "configs/train_segmentation.yaml",
    },
}


def launch_training_job(model_name: str, dry_run: bool = False):

    model_info = MODEL_CONFIG[model_name]
    with open(model_info["config_file"]) as f:
        train_cfg = yaml.safe_load(f)
    sm_cfg = train_cfg.get("sagemaker", {})

    print(f"\n{'='*55}")
    print(f"  SageMaker Training Job — {model_name.upper()}  (SDK V3)")
    print(f"{'='*55}")
    print(f"  Instance     : {sm_cfg.get('instance_type', 'ml.p3.2xlarge')}")
    print(f"  Spot training: {sm_cfg.get('use_spot', True)}")
    print(f"  Max runtime  : {sm_cfg.get('max_run_hours', 6)}h")
    print(f"  S3 bucket    : {S3_BUCKET}")
    print(f"  Entry script : {model_info['entry_script']}")

    if dry_run:
        print("\n  [DRY RUN] Job not submitted.")
        return

    if S3_BUCKET == "your-bucket-name":
        print("\n[ERROR] export SAGEMAKER_S3_BUCKET=visionforge-mahdi")
        sys.exit(1)

    if "YOUR_ACCOUNT_ID" in SAGEMAKER_ROLE:
        print("\n[ERROR] export SAGEMAKER_ROLE_ARN=arn:aws:iam::674095056959:role/SageMakerFullAccess")
        sys.exit(1)

    # ── V3 imports ─────────────────────────────────────────────────────────
    try:
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import (
            SourceCode,
            Compute,
            InputData,
            StoppingCondition,
        )
        from sagemaker.core.training.configs import (
            OutputDataConfig,
            CheckpointConfig,
        )
    except ImportError as e:
        print(f"[ERROR] {e}")
        print("  Run: uv add sagemaker sagemaker-train sagemaker-core")
        sys.exit(1)

    # ── Sync metadata + splits + patches to S3 ─────────────────────────────
    data_s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/processed"
    print("\n  Syncing metadata + splits + patches to S3...")
    for subdir in ["metadata", "splits", "patches"]:
        subprocess.run([
            "aws", "s3", "sync",
            f"data/processed/{subdir}/",
            f"{data_s3_uri}/{subdir}/",
        ], check=True)
    print(f"  Data synced: {data_s3_uri}")

    # ── Config objects ─────────────────────────────────────────────────────
    use_spot     = sm_cfg.get("use_spot", True)
    max_run_sec  = sm_cfg.get("max_run_hours", 6) * 3600
    max_wait_sec = max_run_sec + sm_cfg.get("spot_wait_hours", 1) * 3600

    source_code = SourceCode(
        source_dir="src_package",
        entry_script=model_info["entry_script"],
        requirements="requirements.txt",
        ignore_patterns=[
            ".git", "__pycache__", ".dvc",
            "data/raw", "data/processed/patches",
            ".venv", "reports", "checkpoints",
        ],
    )

    compute = Compute(
        instance_type=sm_cfg.get("instance_type", "ml.p3.2xlarge"),
        instance_count=sm_cfg.get("instance_count", 1),
        volume_size_in_gb=sm_cfg.get("volume_size_gb", 50),
        enable_managed_spot_training=use_spot,
    )

    stopping = StoppingCondition(
        max_runtime_in_seconds=max_run_sec,
        max_wait_time_in_seconds=max_wait_sec if use_spot else None,
    )

    output = OutputDataConfig(
        s3_output_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/output/{model_name}/",
    )

    checkpoint = CheckpointConfig(
        s3_uri=f"s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{model_name}/",
        local_path="/opt/ml/checkpoints",
    )

    # ── Input channels ─────────────────────────────────────────────────────
    input_data = [
        InputData(channel_name="training", data_source=data_s3_uri),
    ]
    if model_name == "anomaly":
        input_data.append(InputData(
            channel_name="mvtec",
            data_source=f"s3://{S3_BUCKET}/{S3_PREFIX}/data/raw/mvtec",
        ))

    # ── ModelTrainer — ALL config in constructor ────────────────────────────
    job_name = f"nulldefect-{model_name}-{str(uuid.uuid4())[:8]}"

    trainer = ModelTrainer(
        training_image=PYTORCH_TRAINING_IMAGE,
        source_code=source_code,
        role=SAGEMAKER_ROLE,
        base_job_name=f"nulldefect-{model_name}",
        compute=compute,
        stopping_condition=stopping,
        output_data_config=output,
        checkpoint_config=checkpoint,
        environment={
            "SAGEMAKER_S3_BUCKET": S3_BUCKET,
        },
    )

    # ── Submit — train() only takes input_data_config + wait ───────────────
    print(f"\n  Submitting job: {job_name}")

    trainer.train(
        input_data_config=input_data,
        wait=False,
    )

    print("\n  [OK] Job submitted!")
    print(f"  Monitor: https://console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/jobs")
    print("\n  Check status:")
    print(
        f"  aws sagemaker list-training-jobs --name-contains nulldefect-{model_name} "
        "--query 'TrainingJobSummaries[0].TrainingJobStatus'"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classifier", "anomaly", "segmentation"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    launch_training_job(args.model, dry_run=args.dry_run)