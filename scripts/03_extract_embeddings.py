import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

# Ensure repository root is in python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.embeddings import FinBERTEmbedder
from src.utils.logger import get_logger

logger = get_logger("03_extract_embeddings")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Dense Financial NLP Embeddings Extraction")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic embeddings for fast pipeline validation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["paths"]["processed_data_dir"]
    cache_dir = cfg["paths"]["embedding_cache_dir"]
    model_name = cfg["nlp"]["model_name"]
    emb_dim = cfg["nlp"]["embedding_dim"]

    df_train = pd.read_parquet(os.path.join(processed_dir, "train_partition.parquet"))
    df_val = pd.read_parquet(os.path.join(processed_dir, "val_partition.parquet"))
    df_test = pd.read_parquet(os.path.join(processed_dir, "test_partition.parquet"))

    if args.mock:
        logger.info(f"Using --mock mode: Generating synthetic {emb_dim}-d dense embeddings for quick testing...")
        rng = np.random.RandomState(42)
        np.save(os.path.join(processed_dir, "train_embeddings.npy"), rng.randn(len(df_train), emb_dim).astype(np.float32))
        np.save(os.path.join(processed_dir, "val_embeddings.npy"), rng.randn(len(df_val), emb_dim).astype(np.float32))
        np.save(os.path.join(processed_dir, "test_embeddings.npy"), rng.randn(len(df_test), emb_dim).astype(np.float32))
        logger.info("Mock embeddings saved successfully.")
        return

    embedder = FinBERTEmbedder(model_name=model_name, cache_dir=cache_dir)
    embedder.extract_partition_embeddings(df_train["aggregated_text"].tolist(), os.path.join(processed_dir, "train_embeddings.npy"), desc="Train")
    embedder.extract_partition_embeddings(df_val["aggregated_text"].tolist(), os.path.join(processed_dir, "val_embeddings.npy"), desc="Validation")
    embedder.extract_partition_embeddings(df_test["aggregated_text"].tolist(), os.path.join(processed_dir, "test_embeddings.npy"), desc="Test")

    logger.info("Phase 3 Complete: Dense 768-d FinBERT embeddings saved.")

if __name__ == "__main__":
    main()
