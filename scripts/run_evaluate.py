# scripts/evaluate.py
"""
Evaluation entry point for MSLG-SPA 2026.

Usage:
    python scripts/evaluate.py --config configs/baseline.yaml --subtask mslg2spa
    python scripts/evaluate.py --config configs/baseline.yaml --subtask spa2mslg
"""

import argparse
import yaml
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch

from src.data.dataset import load_pairs
from src.evaluation.metrics import evaluate_subtask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--subtask", required=True, choices=["mslg2spa", "spa2mslg"])
    return parser.parse_args()


def load_trained_model(checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
        local_files_only=True
    )

    if (checkpoint_dir / "adapter_config.json").exists():
        import json
        adapter_config = json.load(open(checkpoint_dir / "adapter_config.json"))
        base_model_name = adapter_config["base_model_name_or_path"]

        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_dir),
            local_files_only=True
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(checkpoint_dir),
            local_files_only=True
        )

    model.eval()
    return model, tokenizer


def generate_translations(
    model,
    tokenizer,
    sources: list[str],
    max_src_len: int = 128,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> list[str]:
    """
    Generate translations for a list of source sentences.

    Args:
        model:          Trained seq2seq model.
        tokenizer:      Corresponding tokenizer.
        sources:        List of source sentences to translate.
        max_src_len:    Max tokenization length for sources.
        max_new_tokens: Max tokens to generate per translation.
        num_beams:      Beam search width.

    Returns:
        List of translated strings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    translations = []

    # Process one sentence at a time to keep memory usage low
    for source in sources:
        inputs = tokenizer(
            source,
            return_tensors="pt",
            max_length=max_src_len,
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
            )

        translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        translations.append(translation)

    return translations


def main():
    args   = parse_args()
    config = load_config(args.config)

    # ------------------------------------------------------------------ #
    # 1. Load test data
    # ------------------------------------------------------------------ #
    if args.subtask == "mslg2spa":
        test_file = config["data"]["test_mslg2spa"]
        src_col, tgt_col = "mslg", "spa"
    else:
        test_file = config["data"]["test_spa2mslg"]
        src_col, tgt_col = "spa", "mslg"

    df = load_pairs(test_file)
    sources    = df[src_col].tolist()
    references = df[tgt_col].tolist()

    print(f"Loaded {len(df)} test pairs for {args.subtask}")

    # ------------------------------------------------------------------ #
    # 2. Load trained model
    # ------------------------------------------------------------------ #
    checkpoint_dir = Path(config["training"]["output_dir"]) / "final"
    model, tokenizer = load_trained_model(str(checkpoint_dir))
    print(f"Loaded model from {checkpoint_dir}")

    # ------------------------------------------------------------------ #
    # 3. Generate translations
    # ------------------------------------------------------------------ #
    print("Generating translations...")
    predictions = generate_translations(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        max_src_len=config["model"]["max_source_length"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        num_beams=config["generation"]["num_beams"],
    )

    # ------------------------------------------------------------------ #
    # 4. Evaluate
    # ------------------------------------------------------------------ #
    evaluate_subtask(
        sources=sources,
        predictions=predictions,
        references=references,
        subtask=args.subtask,
    )


if __name__ == "__main__":
    main()