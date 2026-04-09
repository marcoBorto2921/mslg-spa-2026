# scripts/ensemble_predict.py
"""
Ensemble prediction for MSLG-SPA 2026.

Loads top-N checkpoints by eval_chrf from trainer_state.json,
generates translations from each, and selects the best translation
per sentence via self-consistency voting (mean chrF against the other N-1).

Usage:
    python scripts/ensemble_predict.py \\
        --config configs/baseline.yaml \\
        --subtask mslg2spa \\
        --checkpoint_dir checkpoints/mslg2spa \\
        --team YourTeam \\
        --solution ensemble3 \\
        --n_checkpoints 3
"""

import argparse
import json
import yaml
from pathlib import Path

from src.data.dataset import load_pairs
from src.evaluation.metrics import compute_chrf
from scripts.run_evaluate import load_trained_model, generate_translations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True,
                        help="Path to YAML config file")
    parser.add_argument("--subtask",        required=True,
                        choices=["mslg2spa", "spa2mslg"])
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing checkpoint-* subdirectories")
    parser.add_argument("--team",           required=True,
                        help="Your team name")
    parser.add_argument("--solution",       required=True,
                        help="Solution label (e.g. ensemble3)")
    parser.add_argument("--n_checkpoints",  type=int, default=3,
                        help="Number of top checkpoints to ensemble (default: 3)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_top_checkpoints(checkpoint_dir: Path, n: int) -> list[Path]:
    """Find top-N checkpoints by eval_chrf from trainer_state.json.

    Reads trainer_state.json from each checkpoint-* subdirectory,
    extracts the eval_chrf logged at that checkpoint's step, sorts
    descending, and returns the top-N paths.

    Args:
        checkpoint_dir: Directory containing checkpoint-* subdirectories.
        n:              Number of top checkpoints to return.

    Returns:
        List of checkpoint Paths sorted by eval_chrf descending.

    Raises:
        ValueError: If no checkpoints with trainer_state.json are found.
    """
    scored: list[tuple[float, Path]] = []

    for ckpt in sorted(checkpoint_dir.glob("checkpoint-*")):
        if not ckpt.is_dir():
            continue
        state_file = ckpt / "trainer_state.json"
        if not state_file.exists():
            continue

        state     = json.load(open(state_file))
        ckpt_step = int(ckpt.name.split("-")[1])

        # Search log_history for the eval_chrf at this checkpoint's step
        chrf_score = None
        for entry in state.get("log_history", []):
            if entry.get("step") == ckpt_step and "eval_chrf" in entry:
                chrf_score = entry["eval_chrf"]
                break

        # Fallback: if this is the best checkpoint, use best_metric
        if chrf_score is None:
            best_ckpt_path = state.get("best_model_checkpoint", "")
            if best_ckpt_path and Path(best_ckpt_path).name == ckpt.name:
                chrf_score = state.get("best_metric")

        if chrf_score is not None:
            scored.append((chrf_score, ckpt))

    if not scored:
        raise ValueError(
            f"No checkpoints with trainer_state.json and eval_chrf found in "
            f"{checkpoint_dir}. Make sure training completed and checkpoints are saved."
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = min(n, len(scored))
    print(f"\nTop-{top_n} checkpoints by eval_chrf:")
    for score, ckpt in scored[:top_n]:
        print(f"  {ckpt.name}  chrF={score:.2f}")

    return [ckpt for _, ckpt in scored[:top_n]]


def self_consistency_vote(predictions_per_model: list[list[str]]) -> list[str]:
    """For each sentence, pick the translation with highest mean chrF against others.

    If all N models agree on a translation, that translation is returned directly.
    Otherwise, each candidate is scored by computing its mean chrF against the
    other N-1 candidates, and the highest-scoring candidate is selected.

    Args:
        predictions_per_model: List of N prediction lists, one per checkpoint.
                               Each inner list has one string per sentence.

    Returns:
        List of selected translations, one per sentence.
    """
    n_models    = len(predictions_per_model)
    n_sentences = len(predictions_per_model[0])
    best        = []

    for i in range(n_sentences):
        candidates = [predictions_per_model[m][i] for m in range(n_models)]

        # Fast path: unanimous agreement
        if len(set(candidates)) == 1:
            best.append(candidates[0])
            continue

        # Score each candidate by mean chrF against the other N-1
        scores = []
        for j, candidate in enumerate(candidates):
            others    = [c for k, c in enumerate(candidates) if k != j]
            mean_chrf = compute_chrf([candidate] * len(others), others) / 100.0
            scores.append(mean_chrf)

        best.append(candidates[scores.index(max(scores))])

    return best


def write_submission(predictions: list[str], output_path: Path) -> None:
    """Write predictions in the official IberLEF submission format.

    Each line: "SystemOutput"\\n
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f'"{pred}"\n')
    print(f"\nSubmission saved to {output_path}  ({len(predictions)} lines)")


def main():
    args   = parse_args()
    config = load_config(args.config)

    # ------------------------------------------------------------------ #
    # 1. Load test sources
    # ------------------------------------------------------------------ #
    if args.subtask == "mslg2spa":
        test_file = config["data"]["test_mslg2spa"]
        src_col   = "mslg"
    else:
        test_file = config["data"]["test_spa2mslg"]
        src_col   = "spa"

    df      = load_pairs(test_file)
    sources = df[src_col].tolist()
    print(f"Loaded {len(sources)} test instances for {args.subtask}")

    # ------------------------------------------------------------------ #
    # 2. Find top-N checkpoints by eval_chrf
    # ------------------------------------------------------------------ #
    checkpoint_dir  = Path(args.checkpoint_dir)
    top_checkpoints = find_top_checkpoints(checkpoint_dir, args.n_checkpoints)

    # ------------------------------------------------------------------ #
    # 3. Generate translations from each checkpoint
    # ------------------------------------------------------------------ #
    all_predictions: list[list[str]] = []
    for ckpt in top_checkpoints:
        print(f"\nLoading {ckpt.name}...")
        model, tokenizer = load_trained_model(str(ckpt))
        preds = generate_translations(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            subtask=args.subtask,
            max_src_len=config["model"]["max_source_length"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            num_beams=config["generation"]["num_beams"],
        )
        all_predictions.append(preds)
        del model  # free GPU memory between checkpoints

    # ------------------------------------------------------------------ #
    # 4. Self-consistency vote
    # ------------------------------------------------------------------ #
    print("\nApplying self-consistency vote...")
    final_predictions = self_consistency_vote(all_predictions)

    # ------------------------------------------------------------------ #
    # 5. Write submission file
    # ------------------------------------------------------------------ #
    filename    = f"{args.team}_{args.solution}_{args.subtask.upper()}.txt"
    output_path = Path("outputs") / filename
    output_path.parent.mkdir(exist_ok=True)
    write_submission(final_predictions, output_path)


if __name__ == "__main__":
    main()
