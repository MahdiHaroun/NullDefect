"""
zenml_stack/setup_stack.py

One-time ZenML stack setup for NullDefect.

Run this ONCE before running any pipelines:
    python zenml_stack/setup_stack.py

What this does:
  1. Initializes ZenML (creates .zen folder)
  2. Registers a local artifact store (stores pipeline outputs)
  3. Registers a local orchestrator
  4. Registers MLflow experiment tracker
  5. Creates and activates the "nulldefect-local" stack

After running, start the dashboard:
    zenml up

Then run the evaluation pipeline:
    python pipelines/evaluation_pipeline.py
"""

import subprocess
import sys


def run(cmd: str):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARN] Command returned {result.returncode} — may already exist, continuing.")


def main():
    print("=" * 55)
    print("  NullDefect — ZenML Stack Setup")
    print("=" * 55)

    # Install ZenML if not present
    print("\n[1/6] Installing ZenML...")
    run("uv add zenml zenml[server] mlflow")

    # Initialize ZenML repo
    print("\n[2/6] Initializing ZenML repository...")
    run("zenml init")

    # Register artifact store — local filesystem
    print("\n[3/6] Registering local artifact store...")
    run(
        "zenml artifact-store register nulldefect_artifact_store "
        "--flavor=local "
        "--path=.zenml_artifacts"
    )

    # Register local orchestrator
    print("\n[4/6] Registering local orchestrator...")
    run(
        "zenml orchestrator register nulldefect_orchestrator "
        "--flavor=local"
    )

    # Register MLflow experiment tracker
    print("\n[5/6] Registering MLflow experiment tracker...")
    run(
        "zenml experiment-tracker register nulldefect_mlflow_tracker "
        "--flavor=mlflow "
        "--tracking_uri=mlruns "
        "--tracking_username=admin "
        "--tracking_password=admin"
    )

    # Create and activate stack
    print("\n[6/6] Creating and activating nulldefect-local stack...")
    run(
        "zenml stack register nulldefect-local "
        "-o nulldefect_orchestrator "
        "-a nulldefect_artifact_store "
        "-e nulldefect_mlflow_tracker"
    )
    run("zenml stack set nulldefect-local")

    print("\n" + "=" * 55)
    print("  Stack setup complete!")
    print("=" * 55)
    print("\n  Start the ZenML dashboard:")
    print("    zenml up")
    print("\n  Run evaluation pipeline:")
    print("    python pipelines/evaluation_pipeline.py")
    print("\n  View stack:")
    print("    zenml stack describe")


if __name__ == "__main__":
    main()
