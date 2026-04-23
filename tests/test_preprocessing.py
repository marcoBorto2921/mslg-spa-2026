# tests/test_preprocessing.py
"""
Unit tests for MSL gloss preprocessing functions.
Run with: pytest tests/test_preprocessing.py -v
"""

from src.data.preprocessing import (
    normalize_msl_glosses,
    apply_hyphen_token,
    preprocess_gloss,
    HYPHEN_TOKEN,
)


# ------------------------------------------------------------------ #
# normalize_msl_glosses
# ------------------------------------------------------------------ #


def test_normalize_removes_dm_prefix():
    assert normalize_msl_glosses("dm-ISABEL TENER CORONA") == "ISABEL TENER CORONA"


def test_normalize_replaces_plus_with_space():
    assert normalize_msl_glosses("MAMÁ+PAPÁ IR CASA") == "MAMÁ PAPÁ IR CASA"


def test_normalize_removes_hash():
    assert normalize_msl_glosses("#OK TODO BIEN") == "OK TODO BIEN"


def test_normalize_keeps_hyphen():
    assert normalize_msl_glosses("LICENCIA-DE-CONDUCIR") == "LICENCIA-DE-CONDUCIR"


def test_normalize_combined():
    result = normalize_msl_glosses("dm-ISABEL #OK MAMÁ+PAPÁ")
    assert result == "ISABEL OK MAMÁ PAPÁ"


def test_normalize_no_double_spaces():
    result = normalize_msl_glosses("dm-A  dm-B")
    assert "  " not in result


def test_normalize_empty_string():
    assert normalize_msl_glosses("") == ""


# ------------------------------------------------------------------ #
# apply_hyphen_token
# ------------------------------------------------------------------ #


def test_apply_hyphen_token_replaces_hyphen():
    result = apply_hyphen_token("LICENCIA-DE-CONDUCIR")
    assert HYPHEN_TOKEN in result
    assert "-" not in result


def test_apply_hyphen_token_handles_no_hyphen():
    assert apply_hyphen_token("YO FELIZ") == "YO FELIZ"


def test_apply_hyphen_token_no_double_spaces():
    result = apply_hyphen_token("A-B")
    assert "  " not in result


# ------------------------------------------------------------------ #
# preprocess_gloss
# ------------------------------------------------------------------ #


def test_preprocess_gloss_both_enabled():
    text = "dm-ISABEL LICENCIA-DE-CONDUCIR"
    result = preprocess_gloss(text, use_hyphen_token=True)
    assert "dm-" not in result
    assert HYPHEN_TOKEN in result
    assert "-" not in result


def test_preprocess_gloss_no_hyphen_token():
    text = "dm-ISABEL LICENCIA-DE-CONDUCIR"
    result = preprocess_gloss(text, use_hyphen_token=False)
    assert "dm-" not in result
    assert HYPHEN_TOKEN not in result
    assert "-" in result


def test_preprocess_gloss_all_annotations():
    text = "dm-JUAN #12 MAMÁ+PAPÁ CASA-GRANDE"
    result = preprocess_gloss(text, use_hyphen_token=True)
    assert "dm-" not in result
    assert "#" not in result
    assert "+" not in result
    assert "-" not in result
    assert HYPHEN_TOKEN in result
