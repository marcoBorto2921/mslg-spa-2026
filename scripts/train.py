# scripts/train.py
"""
Training entry point for MSLG-SPA 2026.

Usage:
    python scripts/train.py --config configs/baseline.yaml --subtask mslg2spa
    python scripts/train.py --config configs/baseline.yaml --subtask spa2mslg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

from src.data.dataset import load_pairs, print_stats, TranslationDataset
from src.models.seq2seq import load_model_and_tokenizer


class BestModelCallback(TrainerCallback):
    """Prints a message on new best and immediately backs up the checkpoint to Drive."""

    def __init__(self, metric_name: str, drive_ckpt_dir: str | None = None) -> None:
        self.metric_name = metric_name
        self.drive_ckpt_dir = Path(drive_ckpt_dir) if drive_ckpt_dir else None
        self.best_score: float = -float("inf")

    def on_save(self, args, state, control, **kwargs) -> None:
        """Triggered right after the Trainer saves a checkpoint."""
        if self.drive_ckpt_dir is None:
            return
        if state.best_model_checkpoint is None:
            return
        import shutil

        src = Path(state.best_model_checkpoint)
        if not src.exists():
            return
        dst = self.drive_ckpt_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [Drive backup] {src.name} → {dst}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        if metrics is None:
            return
        key = f"eval_{self.metric_name}"
        score = metrics.get(key)
        if score is not None and score > self.best_score:
            self.best_score = score
            print(
                f"\n*** NEW BEST — epoch {metrics.get('epoch', '?'):.1f} | "
                f"{self.metric_name} = {score:.4f} *** checkpoint saved\n"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--subtask", required=True, choices=["mslg2spa", "spa2mslg"])
    parser.add_argument(
        "--drive_ckpt_dir", default=None, help="Drive checkpoint dir for live backup"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_compute_metrics(tokenizer, subtask):
    """
    Returns a compute_metrics function for the HuggingFace Trainer.
    Trainer calls this function at the end of each evaluation epoch.
    """
    chrf_metric = evaluate.load("chrf")
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Clip predictions to valid token range before decoding
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

        # Replace -100 (padding) with pad_token_id before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode token ids back to strings
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Compute metrics
        chrf = chrf_metric.compute(
            predictions=decoded_preds, references=[[r] for r in decoded_labels]
        )
        bleu = bleu_metric.compute(
            predictions=decoded_preds, references=[[r] for r in decoded_labels]
        )

        return {
            "chrf": chrf["score"],
            "bleu": bleu["score"],
        }

    return compute_metrics


def main():
    args = parse_args()
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
        lora_target_modules=config["lora"].get("target_modules"),
    )

    # ------------------------------------------------------------------ #
    # 3. Preprocessing (optional)
    # ------------------------------------------------------------------ #
    preprocess_fn = None
    prep_cfg = config.get("preprocessing", {})
    if prep_cfg.get("enabled", False):
        from src.data.preprocessing import preprocess_gloss, add_hyphen_special_token

        use_hyphen = prep_cfg.get("hyphen_special_token", False)
        preprocess_fn = lambda text: preprocess_gloss(text, use_hyphen_token=use_hyphen)
        if use_hyphen:
            add_hyphen_special_token(tokenizer, model)
            print("  [preprocessing] Added [HYPHEN] special token, embeddings resized")
        print(f"  [preprocessing] enabled  |  hyphen_token={use_hyphen}")

    # ------------------------------------------------------------------ #
    # 4. Build datasets
    # ------------------------------------------------------------------ #
    train_dataset = TranslationDataset(
        data=train_df,
        tokenizer=tokenizer,
        subtask=args.subtask,
        max_src_len=config["model"]["max_source_length"],
        max_tgt_len=config["model"]["max_target_length"],
        preprocess_fn=preprocess_fn,
    )
    val_dataset = TranslationDataset(
        data=val_df,
        tokenizer=tokenizer,
        subtask=args.subtask,
        max_src_len=config["model"]["max_source_length"],
        max_tgt_len=config["model"]["max_target_length"],
        preprocess_fn=preprocess_fn,
    )

    # ------------------------------------------------------------------ #
    # 5. Training arguments
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
        save_total_limit=config["training"].get("save_total_limit", 1),
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        predict_with_generate=True,  # needed for seq2seq evaluation
        fp16=config["training"]["fp16"],
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        seed=config["training"]["seed"],
        label_smoothing_factor=config["training"].get("label_smoothing_factor", 0.0),
        # Opt-in: align eval-time decoding with final beam search. Off by default
        # so existing baseline.yaml runs are bit-for-bit identical.
        generation_num_beams=(
            config["generation"].get("num_beams")
            if config["training"].get("eval_with_beam_search", False)
            else None
        ),
        generation_max_length=(
            config["generation"].get("max_new_tokens")
            if config["training"].get("eval_with_beam_search", False)
            else None
        ),
        report_to=config["logging"]["report_to"],
        logging_steps=config["logging"]["logging_steps"],
        run_name=config["logging"].get("run_name"),
    )

    # ------------------------------------------------------------------ #
    # 6. Trainer
    # ------------------------------------------------------------------ #
    callbacks = []
    patience = config["training"].get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    callbacks.append(
        BestModelCallback(
            config["training"]["metric_for_best_model"], args.drive_ckpt_dir
        )
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=make_compute_metrics(tokenizer, args.subtask),
        callbacks=callbacks if callbacks else None,
    )

    # ------------------------------------------------------------------ #
    # 7. Train
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
