# MSLG-SPA 2026 Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add checkpoint ensemble prediction and MSL gloss preprocessing (normalization + hyphen special token) to the MSLG-SPA 2026 pipeline.

**Architecture:** Two independent features. Feature 1 adds a new script that reads trainer checkpoints, ranks by chrF, and combines predictions via self-consistency. Feature 2 adds a preprocessing module applied to MSL glosses before tokenization, controlled by a new config file.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace Transformers + PEFT, evaluate (sacrebleu/chrf), PyYAML

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/data/preprocessing.py` | MSL normalization + hyphen special token |
| Create | `scripts/ensemble_predict.py` | Top-N checkpoint ensemble + self-consistency voting |
| Create | `configs/preprocessing.yaml` | Config for preprocessing experiment |
| Modify | `src/data/dataset.py` | Accept optional `preprocess_fn` in TranslationDataset |
| Modify | `scripts/train.py` | Pass `preprocess_fn` when `preprocessing.enabled` in config |
| Modify | `tests/test_dataset.py` | Tests for preprocessing integration in dataset |
| Create | `tests/test_preprocessing.py` | Unit tests for preprocessing functions |

---

## Task 1: src/data/preprocessing.py

**Files:**
- Create: `src/data/preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1.1: Write failing tests**

```python
# tests/test_preprocessing.py
import pytest
from src.data.preprocessing import (
    normalize_msl_glosses,
    apply_hyphen_token,
    preprocess_gloss,
)


def test_normalize_removes_dm_prefix():
    assert normalize_msl_glosses("dm-ISABEL TENER CORONA") == "ISABEL TENER CORONA"


def test_normalize_replaces_plus_with_space():
    assert normalize_msl_glosses("MAMÁ+PAPÁ IR CASA") == "MAMÁ PAPÁ IR CASA"


def test_normalize_removes_hash():
    assert normalize_msl_glosses("#OK TODO BIEN") == "OK TODO BIEN"


def test_normalize_keeps_hyphen():
    assert normalize_msl_glosses("LICENCIA-DE-CONDUCIR") == "LICENCIA-DE-CONDUCIR"


def test_normalize_combined():
    assert normalize_msl_glosses("dm-ISABEL #OK MAMÁ+PAPÁ") == "ISABEL OK MAMÁ PAPÁ"


def test_apply_hyphen_token_replaces_hyphen():
    result = apply_hyphen_token("LICENCIA-DE-CONDUCIR")
    assert "[HYPHEN]" in result
    assert "-" not in result


def test_apply_hyphen_token_handles_no_hyphen():
    assert apply_hyphen_token("YO FELIZ") == "YO FELIZ"


def test_preprocess_gloss_both_enabled():
    text = "dm-ISABEL LICENCIA-DE-CONDUCIR"
    result = preprocess_gloss(text, use_hyphen_token=True)
    assert "dm-" not in result
    assert "[HYPHEN]" in result


def test_preprocess_gloss_no_hyphen_token():
    text = "dm-ISABEL LICENCIA-DE-CONDUCIR"
    result = preprocess_gloss(text, use_hyphen_token=False)
    assert "dm-" not in result
    assert "[HYPHEN]" not in result
    assert "-" in result
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```
pytest tests/test_preprocessing.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 1.3: Implement src/data/preprocessing.py**

```python
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
      - dm-WORD  -> WORD   (fingerspelling marker, 47 occurrences)
      - WORD+WORD -> WORD WORD  (compound sign, 22 occurrences)
      - #WORD    -> WORD   (number sign, 5 occurrences)
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
    Replacing with a dedicated token prevents the BPE tokenizer from
    fragmenting the compound marker inconsistently.

    Args:
        text: Gloss string (after normalize_msl_glosses).

    Returns:
        Gloss string with - replaced by [HYPHEN].
    """
    return text.replace('-', f' {HYPHEN_TOKEN} ')


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
    text = re.sub(r' +', ' ', text).strip()
    return text


def add_hyphen_special_token(
    tokenizer: PreTrainedTokenizer,
    model,
) -> None:
    """Add [HYPHEN] as a special token and resize model embeddings.

    Must be called before training when use_hyphen_token=True.
    The tokenizer and model must be saved together after this call.

    Args:
        tokenizer: HuggingFace tokenizer to modify in-place.
        model:     Seq2seq model whose embeddings will be resized.
    """
    if HYPHEN_TOKEN not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [HYPHEN_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```
pytest tests/test_preprocessing.py -v
```
Expected: all 9 tests PASS

- [ ] **Step 1.5: Commit**

```bash
git add src/data/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add MSL gloss preprocessing module (normalize + hyphen special token)"
```

---

## Task 2: TranslationDataset accepts preprocess_fn

**Files:**
- Modify: `src/data/dataset.py` (add `preprocess_fn` parameter to `__init__` and `__getitem__`)
- Modify: `tests/test_dataset.py` (add tests for preprocess_fn)

- [ ] **Step 2.1: Add failing test to test_dataset.py**

Add this test to `tests/test_dataset.py`:

```python
def test_dataset_applies_preprocess_fn(sample_df):
    from unittest.mock import MagicMock
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.side_effect = lambda *a, **kw: {
        'input_ids': __import__('torch').zeros(1, 128, dtype=__import__('torch').long),
        'attention_mask': __import__('torch').ones(1, 128, dtype=__import__('torch').long),
    }

    def upper_fn(text: str) -> str:
        return text.upper() + '_PROCESSED'

    dataset = TranslationDataset(
        sample_df, tokenizer, subtask='mslg2spa', preprocess_fn=upper_fn
    )
    # preprocess_fn applied to sources (MSLG side)
    assert dataset.sources[0] == 'YO FELIZ_PROCESSED'
    # targets (SPA side) are never preprocessed
    assert dataset.targets[0] == 'Estoy feliz.'
```

- [ ] **Step 2.2: Run test to confirm it fails**

```
pytest tests/test_dataset.py::test_dataset_applies_preprocess_fn -v
```
Expected: FAIL — `TranslationDataset.__init__` does not accept `preprocess_fn`

- [ ] **Step 2.3: Modify src/data/dataset.py**

Replace the `__init__` signature and body:

```python
def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    subtask: str,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
    preprocess_fn=None,
) -> None:
    assert subtask in ("mslg2spa", "spa2mslg"), \
        "subtask must be 'mslg2spa' or 'spa2mslg'"
    self.tokenizer = tokenizer
    self.max_src_len = max_src_len
    self.max_tgt_len = max_tgt_len

    if subtask == "mslg2spa":
        sources = data["mslg"].tolist()
        self.targets = data["spa"].tolist()
    else:
        sources = data["spa"].tolist()
        self.targets = data["mslg"].tolist()

    # Apply preprocessing to source side only (never to target)
    if preprocess_fn is not None:
        self.sources = [preprocess_fn(s) for s in sources]
    else:
        self.sources = sources
```

- [ ] **Step 2.4: Run all dataset tests**

```
pytest tests/test_dataset.py -v
```
Expected: all 7 tests PASS (6 original + 1 new)

- [ ] **Step 2.5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: TranslationDataset accepts optional preprocess_fn for source side"
```

---

## Task 3: configs/preprocessing.yaml

**Files:**
- Create: `configs/preprocessing.yaml`

- [ ] **Step 3.1: Create the config**

```yaml
# configs/preprocessing.yaml
# Extends baseline with MSL preprocessing experiment.
# Run: python scripts/train.py --config configs/preprocessing.yaml --subtask mslg2spa

model:
  name: facebook/mbart-large-50
  max_source_length: 128
  max_target_length: 128

lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1

data:
  train_file: data/raw/MSLG_SPA_train.txt
  test_mslg2spa: data/raw/test_mslg2spa.tsv
  test_spa2mslg: data/raw/test_spa2mslg.tsv
  processed_dir: data/processed/
  val_split: 0.15

preprocessing:
  enabled: true
  hyphen_special_token: true

training:
  output_dir: checkpoints/preprocessing
  num_train_epochs: 30
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 16
  learning_rate: 5.0e-4
  warmup_steps: 50
  weight_decay: 0.01
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: chrf
  greater_is_better: true
  fp16: True
  seed: 42

generation:
  num_beams: 5
  max_new_tokens: 128

logging:
  report_to: none
  logging_steps: 10
```

- [ ] **Step 3.2: Modify scripts/train.py to wire preprocessing**

In `train.py`, after loading the config and model, add this block just before building datasets:

```python
# Preprocessing (optional — only if config has preprocessing section)
preprocess_fn = None
if config.get("preprocessing", {}).get("enabled", False):
    from src.data.preprocessing import preprocess_gloss, add_hyphen_special_token
    use_hyphen = config["preprocessing"].get("hyphen_special_token", False)
    preprocess_fn = lambda text: preprocess_gloss(text, use_hyphen_token=use_hyphen)
    if use_hyphen:
        add_hyphen_special_token(tokenizer, model)
        print("  [preprocessing] Added [HYPHEN] special token, embeddings resized")
    print(f"  [preprocessing] enabled  |  hyphen_token={use_hyphen}")
```

Then pass `preprocess_fn` to both dataset constructors:

```python
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
```

- [ ] **Step 3.3: Verify train.py still runs dry with baseline config**

```
python scripts/train.py --config configs/baseline.yaml --subtask mslg2spa --help
```
Expected: argparse help printed, no import errors

- [ ] **Step 3.4: Commit**

```bash
git add configs/preprocessing.yaml scripts/train.py
git commit -m "feat: wire MSL preprocessing into train.py via preprocessing config section"
```

---

## Task 4: scripts/ensemble_predict.py

**Files:**
- Create: `scripts/ensemble_predict.py`

- [ ] **Step 4.1: Create the script**

```python
# scripts/ensemble_predict.py
"""
Ensemble prediction for MSLG-SPA 2026.

Loads top-N checkpoints by eval_chrf from trainer_state.json,
generates translations from each, and picks the best translation
per sentence via self-consistency (mean chrF against the other N-1).

Usage:
    python scripts/ensemble_predict.py \\
        --config configs/baseline.yaml \\
        --subtask mslg2spa \\
        --checkpoint_dir checkpoints/mslg2spa \\
        --team YourTeam \\
        --solution ensemble3 \\
        --n_checkpoints 3
"""

import argparse
import json
import yaml
from pathlib import Path

from src.data.dataset import load_pairs
from src.evaluation.metrics import compute_chrf
from scripts.run_evaluate import load_trained_model, generate_translations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--subtask",        required=True, choices=["mslg2spa", "spa2mslg"])
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing checkpoint-* subdirectories")
    parser.add_argument("--team",           required=True)
    parser.add_argument("--solution",       required=True)
    parser.add_argument("--n_checkpoints",  type=int, default=3)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_top_checkpoints(checkpoint_dir: Path, n: int) -> list[Path]:
    """Find the top-N checkpoints by eval_chrf from trainer_state.json.

    Args:
        checkpoint_dir: Directory containing checkpoint-* subdirectories.
        n:              Number of top checkpoints to return.

    Returns:
        List of checkpoint Paths sorted by eval_chrf descending.
    """
    scored: list[tuple[float, Path]] = []

    for ckpt in sorted(checkpoint_dir.glob("checkpoint-*")):
        state_file = ckpt / "trainer_state.json"
        if not state_file.exists():
            continue
        state = json.load(open(state_file))
        # Find the best eval_chrf logged for this checkpoint step
        ckpt_step = int(ckpt.name.split("-")[1])
        chrf_score = None
        for entry in state.get("log_history", []):
            if entry.get("step") == ckpt_step and "eval_chrf" in entry:
                chrf_score = entry["eval_chrf"]
                break
        # Fallback: use best_metric from state
        if chrf_score is None and state.get("best_metric") is not None:
            if str(ckpt) == str(checkpoint_dir / state.get("best_model_checkpoint", "")):
                chrf_score = state["best_metric"]
        if chrf_score is not None:
            scored.append((chrf_score, ckpt))

    if not scored:
        raise ValueError(
            f"No checkpoints with trainer_state.json found in {checkpoint_dir}. "
            "Make sure training has completed and checkpoints are saved."
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [ckpt for _, ckpt in scored[:n]]
    print(f"Top-{n} checkpoints by eval_chrf:")
    for score, ckpt in scored[:n]:
        print(f"  {ckpt.name}  chrF={score:.2f}")
    return top


def self_consistency_vote(predictions_per_model: list[list[str]]) -> list[str]:
    """For each sentence, pick the translation with highest mean chrF against others.

    Args:
        predictions_per_model: List of N prediction lists, one per checkpoint.
                               Each inner list has one translation per sentence.

    Returns:
        List of best translations, one per sentence.
    """
    n_models = len(predictions_per_model)
    n_sentences = len(predictions_per_model[0])
    best_translations = []

    for i in range(n_sentences):
        candidates = [predictions_per_model[m][i] for m in range(n_models)]

        if len(set(candidates)) == 1:
            best_translations.append(candidates[0])
            continue

        # Score each candidate: mean chrF against the other N-1
        scores = []
        for j, candidate in enumerate(candidates):
            others = [c for k, c in enumerate(candidates) if k != j]
            mean_chrf = compute_chrf([candidate] * len(others), others) / 100.0
            scores.append(mean_chrf)

        best_idx = scores.index(max(scores))
        best_translations.append(candidates[best_idx])

    return best_translations


def write_submission(predictions: list[str], output_path: Path) -> None:
    """Write predictions in the official submission format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f'"{pred}"\n')
    print(f"Submission saved to {output_path}  ({len(predictions)} lines)")


def main():
    args   = parse_args()
    config = load_config(args.config)

    # Load test sources
    if args.subtask == "mslg2spa":
        test_file = config["data"]["test_mslg2spa"]
        src_col   = "mslg"
    else:
        test_file = config["data"]["test_spa2mslg"]
        src_col   = "spa"

    df      = load_pairs(test_file)
    sources = df[src_col].tolist()
    print(f"Loaded {len(sources)} test instances for {args.subtask}")

    # Find top-N checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    top_checkpoints = find_top_checkpoints(checkpoint_dir, args.n_checkpoints)

    # Generate translations from each checkpoint
    all_predictions: list[list[str]] = []
    for ckpt in top_checkpoints:
        print(f"\nLoading {ckpt.name}...")
        model, tokenizer = load_trained_model(str(ckpt))
        preds = generate_translations(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            subtask=args.subtask,
            max_src_len=config["model"]["max_source_length"],
            max_new_tokens=config["generation"]["max_new_tokens"],
            num_beams=config["generation"]["num_beams"],
        )
        all_predictions.append(preds)
        del model  # free memory between checkpoints

    # Self-consistency vote
    print("\nApplying self-consistency vote...")
    final_predictions = self_consistency_vote(all_predictions)

    # Write submission
    filename    = f"{args.team}_{args.solution}_{args.subtask.upper()}.txt"
    output_path = Path("outputs") / filename
    output_path.parent.mkdir(exist_ok=True)
    write_submission(final_predictions, output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Smoke test the imports**

```
python -c "from scripts.ensemble_predict import find_top_checkpoints, self_consistency_vote, write_submission; print('OK')"
```
Expected: `OK`

- [ ] **Step 4.3: Commit**

```bash
git add scripts/ensemble_predict.py
git commit -m "feat: add checkpoint ensemble prediction with self-consistency voting"
```

---

## Task 5: Final git push

- [ ] **Step 5.1: Run full test suite**

```
pytest tests/ -v
```
Expected: all tests pass (at minimum test_preprocessing.py + test_dataset.py)

- [ ] **Step 5.2: Push**

```bash
git push origin main
```
