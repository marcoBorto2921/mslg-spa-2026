# scripts/predict.py
"""
Generate official submission files for MSLG-SPA 2026.

Usage:
    python scripts/predict.py --config configs/baseline.yaml --subtask mslg2spa --team YourTeamName --solution baseline
    python scripts/predict.py --config configs/baseline.yaml --subtask spa2mslg --team YourTeamName --solution baseline

Output files follow the official naming convention:
    YourTeamName_baseline_MSLG2SPA.txt
    YourTeamName_baseline_SPA2MSLG.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml

from src.data.dataset import load_pairs
from scripts.run_evaluate import load_trained_model, generate_translations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subtask", required=True, choices=["mslg2spa", "spa2mslg"])
    parser.add_argument("--team", required=True, help="Your team name")
    parser.add_argument(
        "--solution", required=True, help="Solution name (e.g. baseline, lora_r32)"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_submission(
    predictions: list[str],
    output_path: Path,
    ids: list[int] | None = None,
) -> None:
    """Write predictions in the official IberLEF submission format.

    Official format (no ID):
        "SystemOutput"\\n

    Optional format (with instance ID for alignment verification):
        "InstanceIdentifier"\\t"SystemOutput"\\n

    Quotation marks are mandatory. Linux newlines required.
    """
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        if ids is not None:
            for id_, pred in zip(ids, predictions):
                f.write(f'"{id_}"\t"{pred}"\n')
        else:
            for pred in predictions:
                f.write(f'"{pred}"\n')

    print(f"Submission saved to {output_path}")
    print(f"Lines written: {len(predictions)}")


def main():
    args = parse_args()
    config = load_config(args.config)

    # ------------------------------------------------------------------ #
    # 1. Load test data (no references — this is the real test set)
    # ------------------------------------------------------------------ #
    if args.subtask == "mslg2spa":
        test_file = config["data"]["test_mslg2spa"]
        src_col = "mslg"
    else:
        test_file = config["data"]["test_spa2mslg"]
        src_col = "spa"

    import pandas as pd

    raw_df = pd.read_csv(test_file, sep="\t", header=0, encoding="utf-8")
    ids = raw_df["ID"].tolist()
    df = load_pairs(test_file)
    sources = df[src_col].tolist()

    print(f"Loaded {len(sources)} test instances for {args.subtask}")

    # ------------------------------------------------------------------ #
    # 2. Load trained model
    # ------------------------------------------------------------------ #
    checkpoint_dir = Path(config["training"]["output_dir"]) / args.subtask / "final"
    model, tokenizer = load_trained_model(str(checkpoint_dir))

    # ------------------------------------------------------------------ #
    # 3. Generate translations
    # ------------------------------------------------------------------ #
    print("Generating translations...")
    predictions = generate_translations(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        subtask=args.subtask,
        max_src_len=config["model"]["max_source_length"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        num_beams=config["generation"]["num_beams"],
        length_penalty=config["generation"].get("length_penalty", 1.0),
        no_repeat_ngram_size=config["generation"].get("no_repeat_ngram_size", 0),
    )

    # ------------------------------------------------------------------ #
    # 4. Write submission file
    # ------------------------------------------------------------------ #
    # Official naming convention: TeamName_SolutionName_SUBTASK.txt
    filename = f"{args.team}_{args.solution}_{args.subtask.upper()}.txt"
    output_path = Path("outputs") / filename
    output_path.parent.mkdir(exist_ok=True)

    write_submission(predictions, output_path, ids=ids)


if __name__ == "__main__":
    main()
