# src/data/preprocessing.py
"""
MSL gloss preprocessing for MSLG-SPA 2026.

Handles two types of transformations:
  1. Normalization: removes/simplifies rare MSL annotations (dm-, +, #)
  2. Hyphen special token: replaces compound marker - with [HYPHEN]
"""

import re
from transformers import PreTrainedTokenizer


HYPHEN_TOKEN = "[HYPHEN]"


def normalize_msl_glosses(text: str) -> str:
    """Remove rare MSL annotations that appear too infrequently to learn.

    Rules:
      - dm-WORD  -> WORD        (fingerspelling marker, 47 occurrences)
      - WORD+WORD -> WORD WORD  (compound sign, 22 occurrences)
      - #WORD    -> WORD        (number sign, 5 occurrences)
      - Hyphens kept intact (138 occurrences — handled separately)

    Args:
        text: Raw MSL gloss string.

    Returns:
        Normalized gloss string.
    """
    text = re.sub(r'dm-', '', text)
    text = re.sub(r'\+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text


def apply_hyphen_token(text: str) -> str:
    """Replace compound marker hyphen with [HYPHEN] special token.

    In MSL glosses, hyphens always mark compound signs (LICENCIA-DE-CONDUCIR).
    Replacing with a dedicated token prevents BPE from fragmenting the
    compound marker inconsistently across subword units.

    Args:
        text: Gloss string (typically after normalize_msl_glosses).

    Returns:
        Gloss string with - replaced by [HYPHEN].
    """
    return re.sub(r' +', ' ', text.replace('-', f' {HYPHEN_TOKEN} ')).strip()


def preprocess_gloss(text: str, use_hyphen_token: bool = True) -> str:
    """Apply full MSL preprocessing pipeline.

    Args:
        text:              Raw MSL gloss string.
        use_hyphen_token:  If True, replace - with [HYPHEN] special token.

    Returns:
        Preprocessed gloss string.
    """
    text = normalize_msl_glosses(text)
    if use_hyphen_token:
        text = apply_hyphen_token(text)
    return text


def add_hyphen_special_token(
    tokenizer: PreTrainedTokenizer,
    model,
) -> None:
    """Add [HYPHEN] as a special token and resize model embeddings.

    Must be called before training when use_hyphen_token=True.
    The tokenizer and model must both be saved after this call so the
    new vocabulary is persisted.

    Args:
        tokenizer: HuggingFace tokenizer to modify in-place.
        model:     Seq2seq model whose embeddings will be resized.
    """
    if HYPHEN_TOKEN not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [HYPHEN_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
