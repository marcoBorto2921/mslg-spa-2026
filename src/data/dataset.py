# src/data/dataset.py
"""
Dataset loading and preprocessing for MSLG-SPA 2026.

The dataset consists of aligned pairs:
  - MSLG: Mexican Sign Language gloss sequences (e.g., "TÚ LLEGAR TARDE POR QUÉ")
  - SPA:  Spanish sentences (e.g., "¿Por qué llegaste tarde?")

Expected TSV format: two columns, no header.
  Column 0: MSLG gloss sequence
  Column 1: Spanish sentence
"""

import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_pairs(filepath: str | Path) -> pd.DataFrame:
    """
    Load a TSV file of (gloss, spanish) pairs into a DataFrame.
    """
    df = pd.read_csv(filepath, sep="\t", header=0, names=["mslg", "spa"])

    # Basic cleanings
    df = df.dropna() # remoce Nan lines
    df["mslg"] = df["mslg"].str.strip()
    df["spa"] = df["spa"].str.strip()   # remove nitial e final spaces
    df = df[(df["mslg"] != "") & (df["spa"] != "")]  # remove lines with empty string
    df = df.reset_index(drop=True)

    return df


def print_stats(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic corpus statistics.
    Useful to run before training to understand the data.
    """
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Pairs:             {len(df)}")
    print(f"  Avg MSLG tokens:   {df['mslg'].str.split().str.len().mean():.1f}")
    print(f"  Avg SPA tokens:    {df['spa'].str.split().str.len().mean():.1f}")
    print(f"  Max MSLG tokens:   {df['mslg'].str.split().str.len().max()}")
    print(f"  Max SPA tokens:    {df['spa'].str.split().str.len().max()}")
    print(f"  Unique MSLG types: {len(set(' '.join(df['mslg']).split()))}")
    print(f"  Unique SPA types:  {len(set(' '.join(df['spa']).split()))}")


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for sequence-to-sequence translation.

    Handles both subtask directions:
      - mslg2spa: source=mslg, target=spa
      - spa2mslg: source=spa, target=mslg

    Args:
        data:        DataFrame with columns ['mslg', 'spa'].
        tokenizer:   HuggingFace tokenizer.
        subtask:     'mslg2spa' or 'spa2mslg'.
        max_src_len: Max token length for source sequences.
        max_tgt_len: Max token length for target sequences.
    """

    def __init__(   #costruttore
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        subtask: str,
        max_src_len: int = 128,
        max_tgt_len: int = 128,  # lungezze massime
    ) -> None:

        assert subtask in ("mslg2spa", "spa2mslg"), \
            "subtask must be 'mslg2spa' or 'spa2mslg'"

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Assign source/target based on subtask direction
        if subtask == "mslg2spa":
            self.sources = data["mslg"].tolist()
            self.targets = data["spa"].tolist()
        else:
            self.sources = data["spa"].tolist()
            self.targets = data["mslg"].tolist()

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> dict:
        source = self.sources[idx]
        target = self.targets[idx]

        # Tokenize source
        model_inputs = self.tokenizer(
            source,
            max_length=self.max_src_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target
        labels = self.tokenizer(
            text_target=target,
            max_length=self.max_tgt_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Replace padding token id with -100 so it is ignored in the loss
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }