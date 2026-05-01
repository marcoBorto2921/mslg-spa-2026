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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
import evaluate

from src.data.dataset import load_pairs, print_stats, TranslationDataset
from src.models.seq2seq import load_model_and_tokenizer
from src.utils import load_config


class BestModelCallback(TrainerCallback):
    """Prints a message on new best and immediately backs up the checkpoint to Drive.

    Uses a _pending_backup flag set in on_evaluate (where we detect the new best)
    and consumed in on_save (where the checkpoint is guaranteed to be on disk).
    This avoids relying on state.best_model_checkpoint timing inside the Trainer.
    """

    def __init__(
        self, metric_name: str, drive_ckpt_dir: str | None = None, subtask: str = ""
    ) -> None:
        self.metric_name = metric_name
        self.drive_ckpt_dir = Path(drive_ckpt_dir) if drive_ckpt_dir else None
        self.subtask = subtask
        self.best_score: float = -float("inf")
        self._pending_backup: bool = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        if metrics is None:
            return
        score = metrics.get(f"eval_{self.metric_name}")
        if score is not None and score > self.best_score:
            self.best_score = score
            self._pending_backup = True
            print(
                f"\n*** NEW BEST — epoch {metrics.get('epoch', '?'):.1f} | "
                f"{self.metric_name} = {score:.4f} *** checkpoint saved\n"
            )

    def on_save(self, args, state, control, **kwargs) -> None:
        """Triggered right after the Trainer writes a checkpoint to disk."""
        if not self._pending_backup:
            return
        if self.drive_ckpt_dir is None:
            print("  [Drive backup] SKIPPED — --drive_ckpt_dir not set")
            return
        self._pending_backup = False
        import shutil

        src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not src.exists():
            print(f"  [Drive backup] SKIPPED — checkpoint not found at {src.resolve()}")
            return
        # Subtask-scoped Drive path: DRIVE_CKPT/mslg2spa/checkpoint-N
        dst = self.drive_ckpt_dir / self.subtask / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [Drive backup] {src.parent.name}/{src.name} → {dst}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--subtask", required=True, choices=["mslg2spa", "spa2mslg"])
    parser.add_argument(
        "--drive_ckpt_dir", default=None, help="Drive checkpoint dir for live backup"
    )
    return parser.parse_args()


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
    # If real_train_file is set, carve val from real data only to avoid
    # synthetic pairs leaking into validation (data integrity: val must be
    # 100% real, regardless of whether train_file contains augmented data).
    real_file = config["data"].get("real_train_file") or config["data"]["train_file"]
    real_df = load_pairs(real_file)
    print_stats(real_df, name="Real training data")

    train_df, val_df = train_test_split(
        real_df,
        test_size=config["data"]["val_split"],
        random_state=config["training"]["seed"],
    )

    # If an augmented file is provided, append synthetic pairs to train only
    aug_file = config["data"].get("train_file")
    if aug_file and aug_file != real_file:
        if not Path(aug_file).exists():
            raise FileNotFoundError(
                f"Augmented training file not found: {aug_file}\n"
                "Run back_translate.py first to generate it, or switch to baseline.yaml.\n"
                "Example:\n"
                "  python scripts/back_translate.py \\\n"
                "    --config configs/baseline.yaml \\\n"
                "    --spa2mslg_checkpoint checkpoints/baseline/spa2mslg/final \\\n"
                "    --mslg2spa_checkpoint checkpoints/baseline/mslg2spa/final \\\n"
                "    --extract_from_train --output data/processed/augmented_train.tsv \\\n"
                "    --round_trip_threshold 0.0"
            )
        aug_df = load_pairs(aug_file)
        # real_df may not share index with aug_df; deduplicate by content instead
        real_set = set(zip(real_df.iloc[:, 0], real_df.iloc[:, 1]))
        synthetic_df = aug_df[
            ~aug_df.apply(lambda r: (r.iloc[0], r.iloc[1]) in real_set, axis=1)
        ]
        train_df = pd.concat([train_df, synthetic_df], ignore_index=True)
        print(f"  Synthetic pairs appended to train: {len(synthetic_df)}")

    print(f"\n  Train pairs (total): {len(train_df)}")
    print(f"  Val pairs (real only): {len(val_df)}")

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
    # Append subtask name so mslg2spa and spa2mslg don't overwrite each other
    subtask_output_dir = str(Path(config["training"]["output_dir"]) / args.subtask)

    training_args = Seq2SeqTrainingArguments(
        output_dir=subtask_output_dir,
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
        gradient_accumulation_steps=config["training"].get(
            "gradient_accumulation_steps", 1
        ),
        gradient_checkpointing=config["training"].get("gradient_checkpointing", False),
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
            config["training"]["metric_for_best_model"],
            args.drive_ckpt_dir,
            args.subtask,
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
    output_dir = Path(config["training"]["output_dir"]) / args.subtask
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"\nModel saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
