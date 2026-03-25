# tests/test_dataset.py
"""
Basic unit tests for the dataset module.
Run with: pytest tests/
"""

import pytest
import pandas as pd
from src.data.dataset import load_pairs, TranslationDataset


# ------------------------------------------------------------------ #
# Fixtures — reusable test data
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_df():
    """Small DataFrame that mimics the real dataset structure."""
    return pd.DataFrame({
        "mslg": [
            "YO FELIZ",
            "TÚ LLEGAR TARDE POR QUÉ",
            "dm-ISABEL TENER SU CORONA ORO",
            "MAMÁ+PAPÁ CASA IR",
            "#OK TODO BIEN",
        ],
        "spa": [
            "Estoy feliz.",
            "¿Por qué llegaste tarde?",
            "Isabel tiene una corona de oro.",
            "Los padres fueron a casa.",
            "Todo está bien.",
        ]
    })


# ------------------------------------------------------------------ #
# Tests for load_pairs
# ------------------------------------------------------------------ #

def test_load_pairs_returns_dataframe(tmp_path):
    """load_pairs should return a DataFrame with mslg and spa columns."""
    # Create a temporary TSV file
    tsv_content = "MSLG\tSPA\nYO FELIZ\tEstoy feliz.\nTÚ IR\tTú vas.\n"
    tsv_file = tmp_path / "test.tsv"
    tsv_file.write_text(tsv_content, encoding="utf-8")

    df = load_pairs(tsv_file)

    assert isinstance(df, pd.DataFrame)
    assert "mslg" in df.columns
    assert "spa" in df.columns


def test_load_pairs_skips_header(tmp_path):
    """load_pairs should skip the header row."""
    tsv_content = "MSLG\tSPA\nYO FELIZ\tEstoy feliz.\nTÚ IR\tTú vas.\n"
    tsv_file = tmp_path / "test.tsv"
    tsv_file.write_text(tsv_content, encoding="utf-8")

    df = load_pairs(tsv_file)

    # Header row should not appear in data
    assert "MSLG" not in df["mslg"].values
    assert len(df) == 2


def test_load_pairs_drops_empty_rows(tmp_path):
    """load_pairs should remove rows with empty values."""
    tsv_content = "MSLG\tSPA\nYO FELIZ\tEstoy feliz.\n\t\nTÚ IR\tTú vas.\n"
    tsv_file = tmp_path / "test.tsv"
    tsv_file.write_text(tsv_content, encoding="utf-8")

    df = load_pairs(tsv_file)

    assert len(df) == 2


# ------------------------------------------------------------------ #
# Tests for TranslationDataset
# ------------------------------------------------------------------ #

def test_dataset_length(sample_df):
    """Dataset length should match number of pairs."""
    from unittest.mock import MagicMock

    # Mock tokenizer to avoid downloading real model
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.return_value = {
        "input_ids": __import__('torch').zeros(1, 128, dtype=__import__('torch').long),
        "attention_mask": __import__('torch').ones(1, 128, dtype=__import__('torch').long),
    }
    tokenizer.side_effect = lambda *args, **kwargs: {
        "input_ids": __import__('torch').zeros(1, 128, dtype=__import__('torch').long),
        "attention_mask": __import__('torch').ones(1, 128, dtype=__import__('torch').long),
    }

    dataset = TranslationDataset(sample_df, tokenizer, subtask="mslg2spa")
    assert len(dataset) == 5


def test_dataset_subtask_direction(sample_df):
    """Sources and targets should be assigned correctly per subtask."""
    from unittest.mock import MagicMock
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    dataset_mslg2spa = TranslationDataset(sample_df, tokenizer, subtask="mslg2spa")
    dataset_spa2mslg = TranslationDataset(sample_df, tokenizer, subtask="spa2mslg")

    # mslg2spa: source should be glosses
    assert dataset_mslg2spa.sources[0] == "YO FELIZ"
    assert dataset_mslg2spa.targets[0] == "Estoy feliz."

    # spa2mslg: source should be spanish
    assert dataset_spa2mslg.sources[0] == "Estoy feliz."
    assert dataset_spa2mslg.targets[0] == "YO FELIZ"


def test_dataset_invalid_subtask(sample_df):
    """Invalid subtask should raise AssertionError."""
    from unittest.mock import MagicMock
    tokenizer = MagicMock()

    with pytest.raises(AssertionError):
        TranslationDataset(sample_df, tokenizer, subtask="invalid")