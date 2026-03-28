# src/models/seq2seq.py
"""
Model loading utilities for MSLG-SPA 2026.

We use Helsinki-NLP/opus-mt-es-ROMANCE as our baseline model.
LoRA is applied to reduce overfitting on the small training set (489 pairs).
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
):
    """
    Load a seq2seq model and tokenizer, optionally wrapping with LoRA.

    Args:
        model_name:   HuggingFace model identifier.
        use_lora:     Whether to apply LoRA (recommended for this task).
        lora_r:       LoRA rank — higher means more capacity but more parameters.
        lora_alpha:   LoRA scaling factor (typically 2x r).
        lora_dropout: Dropout applied to LoRA layers.

    Returns:
        (model, tokenizer) tuple ready for training or inference.
    """
    # Load tokenizer and model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],  # aggiungi questa riga
            bias="none",
        )

        # Wrap the model with LoRA — freezes base weights,
        # adds small trainable matrices on top
        model = get_peft_model(model, lora_config)

        # Print how many parameters are actually trainable
        model.print_trainable_parameters()

    return model, tokenizer


def count_parameters(model) -> dict[str, int]:
    """
    Return total and trainable parameter counts.
    Useful to verify LoRA is working correctly.

    Returns:
        Dict with keys 'total' and 'trainable'.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}