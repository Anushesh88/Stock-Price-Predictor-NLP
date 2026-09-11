import argparse
import os
import subprocess
import sys

# Ensure repository root is in python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger

logger = get_logger("run_pipeline")

def run_step(cmd: list, desc: str):
    logger.info(f"\n{'='*60}\n>>> STARTING: {desc}\n{'='*60}")
    res = subprocess.run([sys.executable] + cmd)
    if res.returncode != 0:
        logger.error(f"Step '{desc}' failed with return code {res.returncode}.")
        sys.exit(res.returncode)
    logger.info(f">>> COMPLETED: {desc}\n")

def main():
    parser = argparse.ArgumentParser(description="End-to-End Multimodal Stock Predictor Pipeline Runner")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--mock-embeddings", action="store_true", help="Use synthetic mock embeddings for rapid testing")
    parser.add_argument("--sample", action="store_true", help="Use synthetic sample market data instead of live download")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--skip-download", action="store_true", help="Skip Phase 1 data download if already completed")
    args = parser.parse_args()

    # Step 1: Download and Align
    if not args.skip_download:
        p1_cmd = ["scripts/01_download_and_align.py", "--config", args.config]
        if args.sample:
            p1_cmd.append("--sample")
        run_step(p1_cmd, "Phase 1: Ingestion & Trading Calendar Alignment")

    # Step 2: Feature Engineering & Partitioning
    run_step(["scripts/02_engineer_features.py", "--config", args.config], "Phase 2: Technical Indicators, Labeling & Partitioning")

    # Step 3: Embeddings Extraction
    emb_cmd = ["scripts/03_extract_embeddings.py", "--config", args.config]
    if args.mock_embeddings:
        emb_cmd.append("--mock")
    run_step(emb_cmd, "Phase 3: Dense FinBERT Text Embeddings Extraction")

    # Step 4: Training
    train_cmd = ["scripts/04_train.py", "--config", args.config]
    if args.epochs:
        train_cmd.extend(["--epochs", str(args.epochs)])
    run_step(train_cmd, "Phase 4: Multimodal Model Training with Focal Loss")

    # Step 5: Evaluation & Backtesting
    run_step(["scripts/05_backtest.py", "--config", args.config], "Phase 5: Out-of-Sample Evaluation & Financial Backtest")

    logger.info("\n" + "#" * 60)
    logger.info("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    logger.info("#" * 60)

if __name__ == "__main__":
    main()
