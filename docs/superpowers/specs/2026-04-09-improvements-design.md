# Design: MSLG-SPA 2026 Improvements

**Date:** 2026-04-09
**Status:** Approved
**Scope:** Checkpoint ensemble + MSL preprocessing (LLM few-shot deferred)

---

## Feature 1 — Checkpoint Ensemble

**What:** Generate translations from top-N checkpoints by `eval_chrf`, combine via self-consistency voting.

**Files:**
- `scripts/ensemble_predict.py` (new)

**Logic:**
1. Scan `checkpoints/<subtask>/checkpoint-*/trainer_state.json`
2. Extract `eval_chrf` from `log_history` for each checkpoint
3. Sort descending, take top-N (default 3)
4. Load each checkpoint with `load_trained_model()`
5. For each source sentence, generate N translations
6. Pick the translation with highest mean chrF against the other N-1 (self-consistency)
7. Write submission file: `outputs/Team_solution_ENSEMBLE_SUBTASK.txt`

**Interface:**
```
python scripts/ensemble_predict.py \
  --config configs/baseline.yaml \
  --subtask mslg2spa \
  --team YourTeam \
  --solution ensemble3 \
  --n_checkpoints 3
```

---

## Feature 2 — MSL Preprocessing

**What:** Normalize rare MSL annotations + add `[HYPHEN]` special token for compound signs.

**Files:**
- `src/data/preprocessing.py` (new)
- `configs/preprocessing.yaml` (new — extends baseline, does not modify it)
- `scripts/train.py` (minor: pass preprocessing functions to dataset)

**Normalization rules:**
| Pattern | Example | Result |
|---|---|---|
| `dm-` prefix | `dm-ISABEL` | `ISABEL` |
| `+` compound | `MAMÁ+PAPÁ` | `MAMÁ PAPÁ` |
| `#` number sign | `#OK` | `OK` |
| `-` compound (special token) | `LICENCIA-DE-CONDUCIR` | `LICENCIA [HYPHEN] DE [HYPHEN] CONDUCIR` |

**API:**
```python
normalize_msl_glosses(text: str) -> str          # dm-, +, # only
add_hyphen_special_token(tokenizer, model)        # adds [HYPHEN], resizes embeddings
apply_hyphen_token(text: str) -> str              # replaces - with [HYPHEN]
preprocess_gloss(text: str, use_hyphen_token: bool) -> str  # combined entry point
```

**Config:**
```yaml
# configs/preprocessing.yaml
preprocessing:
  enabled: true
  hyphen_special_token: true
```

**Dataset integration:** `TranslationDataset` accepts optional `preprocess_fn: Callable | None`. When set, applies to source sequences (MSLG side only — Spanish side is never preprocessed).

---

## What is NOT changing

- `configs/baseline.yaml` — untouched, baseline remains reproducible
- `src/evaluation/metrics.py` — untouched
- Existing training results — untouched
