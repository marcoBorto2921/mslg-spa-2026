# scripts/train.py
"""
Training entry point for MSLG-SPA 2026.

Usage:
    python scripts/train.py --config configs/baseline.yaml --subtask mslg2spa
    python scripts/train.py --config configs/baseline.yaml --subtask spa2mslg
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate

from src.data.dataset import load_pairs, print_stats, TranslationDataset
from src.models.seq2seq import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--subtask", required=True, choices=["mslg2spa", "spa2mslg"])
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_compute_metrics(tokenizer, subtask):
    """
    Returns a compute_metrics function for the HuggingFace Trainer.
    Trainer calls this function at the end of each evaluation epoch.
    """
    chrf_metric   = evaluate.load("chrf")
    bleu_metric   = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 (padding) with pad_token_id before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode token ids back to strings
        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)

        # Strip whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Compute metrics
        chrf = chrf_metric.compute(
            predictions=decoded_preds,
            references=[[r] for r in decoded_labels]
        )
        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[r] for r in decoded_labels]
        )

        return {
            "chrf": chrf["score"],
            "bleu": bleu["score"],
        }

    return compute_metrics


def main():
    args   = parse_args()
    config = load_config(args.config)

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    df = load_pairs(config["data"]["train_file"])
    print_stats(df, name="Training data")

    # Split into train / validation
    train_df, val_df = train_test_split(
        df,
        test_size=config["data"]["val_split"],
        random_state=config["training"]["seed"],
    )
    print(f"\n  Train pairs: {len(train_df)}")
    print(f"  Val pairs:   {len(val_df)}")

    # ------------------------------------------------------------------ #
    # 2. Load model and tokenizer
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["name"],
        use_lora=config["lora"]["enabled"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
    )

    # ------------------------------------------------------------------ #
    # 3. Build datasets
    # ------------------------------------------------------------------ #
    train_dataset = TranslationDataset(
        data=train_df,
        tokenizer=tokenizer,
        subtask=args.subtask,
        max_src_len=config["model"]["max_source_length"],
        max_tgt_len=config["model"]["max_target_length"],
    )
    val_dataset = TranslationDataset(
        data=val_df,
        tokenizer=tokenizer,
        subtask=args.subtask,
        max_src_len=config["model"]["max_source_length"],
        max_tgt_len=config["model"]["max_target_length"],
    )

    # ------------------------------------------------------------------ #
    # 4. Training arguments
    # ------------------------------------------------------------------ #
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        predict_with_generate=True,  # needed for seq2seq evaluation
        fp16=config["training"]["fp16"],
        seed=config["training"]["seed"],
        report_to=config["logging"]["report_to"],
        logging_steps=config["logging"]["logging_steps"],
    )

    # ------------------------------------------------------------------ #
    # 5. Trainer
    # ------------------------------------------------------------------ #
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=make_compute_metrics(tokenizer, args.subtask),
    )

    # ------------------------------------------------------------------ #
    # 6. Train
    # ------------------------------------------------------------------ #
    print(f"\nStarting training — subtask: {args.subtask}")
    trainer.train()

    # Save final model
    output_dir = Path(config["training"]["output_dir"])
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"\nModel saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()