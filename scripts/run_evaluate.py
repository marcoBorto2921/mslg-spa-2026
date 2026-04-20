# scripts/evaluate.py
"""
Evaluation entry point for MSLG-SPA 2026.

Usage:
    python scripts/evaluate.py --config configs/baseline.yaml --subtask mslg2spa
    python scripts/evaluate.py --config configs/baseline.yaml --subtask spa2mslg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

from src.data.dataset import load_pairs
from src.evaluation.metrics import evaluate_subtask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--subtask", required=True, choices=["mslg2spa", "spa2mslg"])
    parser.add_argument(
        "--comet",
        action="store_true",
        help="Compute COMET (mslg2spa only). Requires unbabel-comet.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_trained_model(checkpoint_dir: str):
    from transformers import MBart50Tokenizer

    checkpoint_dir = Path(checkpoint_dir)

    # Load tokenizer from base model — local checkpoint may lack tokenizer files
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
    # mBART-50 requires src_lang/tgt_lang for text_target tokenization
    tokenizer.src_lang = "es_XX"
    tokenizer.tgt_lang = "es_XX"

    if (checkpoint_dir / "adapter_config.json").exists():
        import json

        adapter_config = json.load(open(checkpoint_dir / "adapter_config.json"))
        base_model_name = adapter_config["base_model_name_or_path"]

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name, local_files_only=False
        )
        model = PeftModel.from_pretrained(
            base_model, str(checkpoint_dir), local_files_only=False
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(checkpoint_dir), local_files_only=False
        )
    model.eval()
    return model, tokenizer


def generate_translations(
    model,
    tokenizer,
    sources: list[str],
    subtask: str = "mslg2spa",
    max_src_len: int = 128,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> list[str]:
    """
    Generate translations for a list of source sentences.

    Args:
        model:                Trained seq2seq model.
        tokenizer:            Corresponding tokenizer.
        sources:              List of source sentences to translate.
        subtask:              'mslg2spa' or 'spa2mslg' — determines target language token.
        max_src_len:          Max tokenization length for sources.
        max_new_tokens:       Max tokens to generate per translation.
        num_beams:             Beam search width.
        length_penalty:       Exponential penalty to beam scores; >1 favors longer
                              outputs, <1 favors shorter. 1.0 = neutral (HF default).
        no_repeat_ngram_size: If >0, block repetition of n-grams of this size in
                              generation. Typical: 0 (off) or 3. Useful for short
                              glosses where mBART can loop.

    Returns:
        List of translated strings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # mBART requires forced_bos_token_id to set the target language
    # MSLG2SPA: target is Spanish; SPA2MSLG: target is Spanish (MSL has no mBART code)
    forced_bos_token_id = tokenizer.lang_code_to_id.get("es_XX")

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
                forced_bos_token_id=forced_bos_token_id,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        translations.append(translation)

    return translations


def main():
    args = parse_args()
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
    sources = df[src_col].tolist()
    references = df[tgt_col].tolist()

    print(f"Loaded {len(df)} test pairs for {args.subtask}")

    # ------------------------------------------------------------------ #
    # 2. Load trained model
    # ------------------------------------------------------------------ #
    checkpoint_dir = Path(config["training"]["output_dir"]) / args.subtask / "final"
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
        subtask=args.subtask,
        max_src_len=config["model"]["max_source_length"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        num_beams=config["generation"]["num_beams"],
        length_penalty=config["generation"].get("length_penalty", 1.0),
        no_repeat_ngram_size=config["generation"].get("no_repeat_ngram_size", 0),
    )

    # ------------------------------------------------------------------ #
    # 4. Evaluate
    # ------------------------------------------------------------------ #
    evaluate_subtask(
        sources=sources,
        predictions=predictions,
        references=references,
        subtask=args.subtask,
        include_comet=args.comet,
    )


if __name__ == "__main__":
    main()
