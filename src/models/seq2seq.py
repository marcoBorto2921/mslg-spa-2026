# src/models/seq2seq.py
"""
Model loading utilities for MSLG-SPA 2026.

We use Helsinki-NLP/opus-mt-es-ROMANCE as our baseline model.
LoRA is applied to reduce overfitting on the small training set (489 pairs).
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def load_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: list[str] | None = None,
):
    """
    Load a seq2seq model and tokenizer, optionally wrapping with LoRA.

    Args:
        model_name:          HuggingFace model identifier.
        use_lora:            Whether to apply LoRA (recommended for this task).
        lora_r:              LoRA rank — higher means more capacity but more parameters.
        lora_alpha:          LoRA scaling factor (typically 2x r).
        lora_dropout:        Dropout applied to LoRA layers.
        lora_target_modules: List of attention projection names to adapt with LoRA.
                             Defaults to ["q_proj", "v_proj"]. For mBART, valid
                             attention names are q_proj/k_proj/v_proj/out_proj;
                             FFN names are fc1/fc2.

    Returns:
        (model, tokenizer) tuple ready for training or inference.
    """
    # Load tokenizer and model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # mBART-50 requires src_lang/tgt_lang for text_target tokenization
    if hasattr(tokenizer, "src_lang") and tokenizer.src_lang is None:
        tokenizer.src_lang = "es_XX"
    if hasattr(tokenizer, "tgt_lang") and tokenizer.tgt_lang is None:
        tokenizer.tgt_lang = "es_XX"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = (
            list(lora_target_modules)
            if lora_target_modules is not None
            else list(DEFAULT_LORA_TARGET_MODULES)
        )
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        print(f"  [LoRA] target_modules = {target_modules}")

        # Wrap the model with LoRA — freezes base weights,
        # adds small trainable matrices on top
        model = get_peft_model(model, lora_config)

        # Print how many parameters are actually trainable
        model.print_trainable_parameters()

    return model, tokenizer
