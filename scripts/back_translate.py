"""
Back-translation data augmentation for MSLG-SPA 2026.

Pipeline:
  1. Collect Spanish sentences (external file + optionally SPA column from training file)
  2. Translate SPA → MSLG using trained SPA2MSLG model
  3. Optional round-trip consistency filter
  4. Save augmented dataset (original + synthetic pairs) to TSV

Usage:
    python scripts/back_translate.py \
        --config configs/baseline.yaml \
        --spa2mslg_checkpoint checkpoints/baseline/spa2mslg/final \
        --mslg2spa_checkpoint checkpoints/baseline/mslg2spa/final \
        --output data/processed/augmented_train.tsv \
        --extract_from_train \
        --round_trip_threshold 0.0

Notes:
  - --extract_from_train pulls all SPA sentences from config["data"]["train_file"]
    and combines them with --spa_file (if provided). Combined unique sentences are
    used as the BT source pool.
  - --round_trip_threshold 0.0 keeps all synthetic pairs (recommended when models
    are still weak; noisy pairs still provide useful signal).
  - Output TSV has header row: mslg<TAB>spa. Compatible with load_pairs().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd

from src.data.dataset import load_pairs
from src.utils import load_config
from src.evaluation.metrics import compute_chrf
from scripts.run_evaluate import load_trained_model, generate_translations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--spa_file",
        default=None,
        help="Plain text file with one Spanish sentence per line",
    )
    parser.add_argument(
        "--extract_from_train",
        action="store_true",
        help="Also extract SPA sentences from config train_file",
    )
    parser.add_argument(
        "--output", required=True, help="Output TSV path for augmented training data"
    )
    parser.add_argument(
        "--mslg2spa_checkpoint",
        required=True,
        help="Path to trained MSLG2SPA model checkpoint (for round-trip)",
    )
    parser.add_argument(
        "--spa2mslg_checkpoint",
        required=True,
        help="Path to trained SPA2MSLG model checkpoint",
    )
    parser.add_argument(
        "--round_trip_threshold",
        type=float,
        default=0.0,
        help="Min chrF (0–100) for round-trip filter. 0.0 = keep all.",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Cap on total BT source sentences (default: all)",
    )
    parser.add_argument(
        "--direction",
        choices=["spa2mslg", "mslg2spa"],
        default="spa2mslg",
        help=(
            "spa2mslg (default): SPA→MSLG synthetic pairs, helps MSLG2SPA training.\n"
            "mslg2spa: MSLG→SPA synthetic pairs, helps SPA2MSLG training."
        ),
    )
    return parser.parse_args()


def collect_spanish_sentences(
    spa_file: str | None,
    extract_from_train: bool,
    train_file: str,
    max_sentences: int | None,
) -> list[str]:
    """
    Collect unique Spanish sentences from one or both sources:
      1. Plain text file (--spa_file), one sentence per line
      2. SPA column of training TSV (--extract_from_train)

    Args:
        spa_file:            Path to plain-text Spanish file, or None.
        extract_from_train:  If True, load SPA column from training TSV.
        train_file:          Path to training TSV (used when extract_from_train=True).
        max_sentences:       Optional cap on total sentences returned.

    Returns:
        Deduplicated list of Spanish sentences.
    """
    sentences: list[str] = []

    if spa_file is not None:
        with open(spa_file, "r", encoding="utf-8") as f:
            file_sents = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(file_sents)} sentences from {spa_file}")
        sentences.extend(file_sents)

    if extract_from_train:
        df = load_pairs(train_file)
        train_sents = df["spa"].tolist()
        print(f"  Extracted {len(train_sents)} SPA sentences from training file")
        sentences.extend(train_sents)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    if max_sentences is not None:
        unique = unique[:max_sentences]

    print(f"  Total unique BT source sentences: {len(unique)}")
    return unique


def round_trip_filter(
    spa_originals: list[str],
    synthetic_glosses: list[str],
    mslg2spa_model,
    mslg2spa_tokenizer,
    threshold: float,
    max_src_len: int,
    max_new_tokens: int,
    num_beams: int,
) -> list[tuple[str, str]]:
    """
    Filter synthetic pairs by round-trip consistency.

    For each (spa_original, synthetic_gloss) pair:
      1. Translate synthetic_gloss → Spanish via MSLG2SPA
      2. Compute chrF(original_spa, reconstructed_spa)
      3. Keep if chrF >= threshold

    Args:
        spa_originals:      Original Spanish sentences used as BT source.
        synthetic_glosses:  Synthetic MSL glosses from SPA2MSLG.
        mslg2spa_model:     Trained MSLG2SPA model.
        mslg2spa_tokenizer: Corresponding tokenizer.
        threshold:          Minimum chrF score (0–100) to keep a pair.
        max_src_len:        Max tokenization length.
        max_new_tokens:     Max generation tokens.
        num_beams:          Beam width.

    Returns:
        List of (mslg, spa) pairs that passed the filter.
    """
    print("Running round-trip filter...")

    reconstructed_spa = generate_translations(
        model=mslg2spa_model,
        tokenizer=mslg2spa_tokenizer,
        sources=synthetic_glosses,
        subtask="mslg2spa",
        max_src_len=max_src_len,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    kept_pairs: list[tuple[str, str]] = []
    filtered_count = 0

    for orig, gloss, recon in zip(spa_originals, synthetic_glosses, reconstructed_spa):
        score = compute_chrf([recon], [orig])  # returns 0–100
        if score >= threshold:
            kept_pairs.append((gloss, orig))
        else:
            filtered_count += 1

    print(
        f"  Round-trip: kept {len(kept_pairs)}, "
        f"filtered {filtered_count} (threshold={threshold})"
    )
    return kept_pairs


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    max_src_len = config["model"]["max_source_length"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    num_beams = config["generation"]["num_beams"]
    real_train_file = (
        config["data"].get("real_train_file") or config["data"]["train_file"]
    )

    if args.direction == "mslg2spa":
        _run_reverse_bt(
            args, config, max_src_len, max_new_tokens, num_beams, real_train_file
        )
    else:
        _run_forward_bt(
            args, config, max_src_len, max_new_tokens, num_beams, real_train_file
        )


def _run_forward_bt(
    args, config, max_src_len, max_new_tokens, num_beams, real_train_file
):
    """SPA→MSLG: synthetic (mslg_synthetic, spa_real) pairs — helps MSLG2SPA training."""
    train_file = config["data"]["train_file"]

    # ------------------------------------------------------------------ #
    # 1. Collect Spanish source sentences
    # ------------------------------------------------------------------ #
    print("\n[1] Collecting Spanish sentences...")
    spa_sentences = collect_spanish_sentences(
        spa_file=args.spa_file,
        extract_from_train=args.extract_from_train,
        train_file=train_file,
        max_sentences=args.max_sentences,
    )

    if not spa_sentences:
        raise ValueError(
            "No Spanish sentences found. "
            "Provide --spa_file or use --extract_from_train."
        )

    # ------------------------------------------------------------------ #
    # 2. Load SPA2MSLG model and generate synthetic glosses
    # ------------------------------------------------------------------ #
    print("\n[2] Loading SPA2MSLG model...")
    spa2mslg_model, spa2mslg_tokenizer = load_trained_model(args.spa2mslg_checkpoint)

    print("Generating synthetic MSL glosses...")
    synthetic_glosses = generate_translations(
        model=spa2mslg_model,
        tokenizer=spa2mslg_tokenizer,
        sources=spa_sentences,
        subtask="spa2mslg",
        max_src_len=max_src_len,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    print(f"  Generated {len(synthetic_glosses)} synthetic glosses")
    print("\n  Examples:")
    for spa, gloss in zip(spa_sentences[:5], synthetic_glosses[:5]):
        print(f"    SPA:  {spa}")
        print(f"    MSLG: {gloss}")
        print()

    # ------------------------------------------------------------------ #
    # 3. Round-trip filter (skip if threshold == 0.0)
    # ------------------------------------------------------------------ #
    if args.round_trip_threshold > 0.0:
        print("\n[3] Loading MSLG2SPA model for round-trip filter...")
        mslg2spa_model, mslg2spa_tokenizer = load_trained_model(
            args.mslg2spa_checkpoint
        )

        kept_pairs = round_trip_filter(
            spa_originals=spa_sentences,
            synthetic_glosses=synthetic_glosses,
            mslg2spa_model=mslg2spa_model,
            mslg2spa_tokenizer=mslg2spa_tokenizer,
            threshold=args.round_trip_threshold,
            max_src_len=max_src_len,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    else:
        print("\n[3] Round-trip filter skipped (threshold=0.0 — keeping all pairs)")
        kept_pairs = list(zip(synthetic_glosses, spa_sentences))

    # ------------------------------------------------------------------ #
    # 4. Combine with original training data and save
    # ------------------------------------------------------------------ #
    print("\n[4] Combining with original training data...")
    original_df = load_pairs(real_train_file)
    synthetic_df = pd.DataFrame(kept_pairs, columns=["mslg", "spa"])

    print(f"  Original pairs:  {len(original_df)}")
    print(f"  Synthetic pairs: {len(synthetic_df)}")

    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    print(f"  Total augmented: {len(augmented_df)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nAugmented dataset saved to {output_path}")


def _run_reverse_bt(
    args, config, max_src_len, max_new_tokens, num_beams, real_train_file
):
    """MSLG→SPA: synthetic (mslg_real, spa_synthetic) pairs — helps SPA2MSLG training."""

    # ------------------------------------------------------------------ #
    # 1. Collect real MSLG glosses from training data
    # ------------------------------------------------------------------ #
    print("\n[1] Collecting real MSLG glosses from training data...")
    original_df = load_pairs(real_train_file)
    mslg_sentences = original_df["mslg"].tolist()
    if args.max_sentences is not None:
        mslg_sentences = mslg_sentences[: args.max_sentences]
    print(f"  Total MSLG source sentences: {len(mslg_sentences)}")

    # ------------------------------------------------------------------ #
    # 2. Load MSLG2SPA model and generate synthetic Spanish
    # ------------------------------------------------------------------ #
    print("\n[2] Loading MSLG2SPA model...")
    mslg2spa_model, mslg2spa_tokenizer = load_trained_model(args.mslg2spa_checkpoint)

    print("Generating synthetic Spanish sentences...")
    synthetic_spa = generate_translations(
        model=mslg2spa_model,
        tokenizer=mslg2spa_tokenizer,
        sources=mslg_sentences,
        subtask="mslg2spa",
        max_src_len=max_src_len,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    print(f"  Generated {len(synthetic_spa)} synthetic Spanish sentences")
    print("\n  Examples:")
    for gloss, spa in zip(mslg_sentences[:5], synthetic_spa[:5]):
        print(f"    MSLG: {gloss}")
        print(f"    SPA:  {spa}")
        print()

    # ------------------------------------------------------------------ #
    # 3. Round-trip filter skipped for reverse BT (threshold always 0.0)
    # ------------------------------------------------------------------ #
    print("\n[3] Round-trip filter skipped for reverse BT — keeping all pairs")
    kept_pairs = list(zip(mslg_sentences, synthetic_spa))  # (mslg_real, spa_synthetic)

    # ------------------------------------------------------------------ #
    # 4. Combine with original training data and save
    # ------------------------------------------------------------------ #
    print("\n[4] Combining with original training data...")
    synthetic_df = pd.DataFrame(kept_pairs, columns=["mslg", "spa"])

    print(f"  Original pairs:  {len(original_df)}")
    print(f"  Synthetic pairs: {len(synthetic_df)}")

    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    print(f"  Total augmented: {len(augmented_df)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nReverse-BT augmented dataset saved to {output_path}")


if __name__ == "__main__":
    main()
