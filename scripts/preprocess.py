# scripts/preprocess.py
"""
Preprocessing script for MSLG-SPA 2026.

Validates the dataset, prints statistics, and saves
cleaned versions to data/processed/.

Usage:
    python scripts/preprocess.py --config configs/baseline.yaml
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.data.dataset import load_pairs, print_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_pairs(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the dataset.
    Prints warnings if anything looks wrong.
    """
    issues = 0

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"[WARNING] {duplicates} duplicate rows found")
        issues += 1

    # Check for very long sequences
    long_mslg = (df['mslg'].str.split().str.len() > 50).sum()
    long_spa  = (df['spa'].str.split().str.len() > 50).sum()
    if long_mslg > 0:
        print(f"[WARNING] {long_mslg} MSLG sequences longer than 50 tokens")
        issues += 1
    if long_spa > 0:
        print(f"[WARNING] {long_spa} SPA sequences longer than 50 tokens")
        issues += 1

    # Check for mismatched lengths (very short gloss, very long spanish)
    df['mslg_len'] = df['mslg'].str.split().str.len()
    df['spa_len']  = df['spa'].str.split().str.len()
    ratio = df['spa_len'] / df['mslg_len']
    extreme = (ratio > 5).sum()
    if extreme > 0:
        print(f"[WARNING] {extreme} pairs with SPA/MSLG length ratio > 5")
        issues += 1

    if issues == 0:
        print("[OK] No issues found in dataset")


def main():
    args   = parse_args()
    config = load_config(args.config)

    # Load raw data
    train_file = config["data"]["train_file"]
    df = load_pairs(train_file)
    print_stats(df, "Raw training data")

    # Validate
    print("\nRunning validation checks...")
    validate_pairs(df)

    # Save cleaned version to processed/
    output_dir = Path(config["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "train_clean.tsv"
    df[["mslg", "spa"]].to_csv(output_path, sep="\t", index=False)
    print(f"\nClean dataset saved to {output_path}")
    print(f"Total pairs: {len(df)}")


if __name__ == "__main__":
    main()