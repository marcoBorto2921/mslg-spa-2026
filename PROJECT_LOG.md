# PROJECT_LOG — MSLG-SPA 2026 (IberLEF)

> last_updated: 2026-04-09 | session: 2
> status: IN PROGRESS

---

## Project Context
Bidirectional NLP pipeline for MSL gloss <-> Spanish translation.
IberLEF 2026 shared task — submission deadline: May 2026.
Repo: github.com/marcoBorto2921/mslg-spa-2026

**Tech Stack:** Python 3.11, PyTorch 2.x, HuggingFace Transformers 5.x,
mBART-large-50, LoRA via PEFT 0.10+, training on Colab T4 (16GB)

---

## Current Status
Both subtasks trained and evaluated. Back-translation augmentation implemented and
confirmed effective (+4.06 chrF on MSLG2SPA). Official test set not yet received
(expected within days). Submission pipeline (predict.py) ready.

**Last session (2026-04-09):** Fixed 3 critical bugs (load_config missing, forced_bos_token_id missing,
Italian comments). Added subtask param to generate_translations. num_beams 4→5.

---

## Decision Log

| Date | Decision | Rationale | Discarded Alternative |
|------|----------|-----------|----------------------|
| — | mBART-large-50 | Native seq2seq, Spanish pretraining, multilingual robustness | Helsinki-NLP (wrong direction), MT5 (complex setup), decoder-only (wrong architecture) |
| — | LoRA r=16, alpha=32 | Quality/memory balance on T4 (only 2.36M trainable params) | Full fine-tuning (OOM) |
| — | chrF as best-model metric | More robust than BLEU on short sequences and small vocab | BLEU (too sensitive to exact word choice) |
| — | 85%/15% hold-out split | Simplicity over k-fold; HF Trainer doesn't natively support k-fold | StratifiedKFold (complex, marginal benefit here) |
| — | Back-translation threshold=0.0 | threshold=0.1 kept only 7/100 — too strict; noisy pairs still help | threshold=0.1 (7 pairs kept — statistically useless) |

---

## Failed Attempts

- **Back-translation round-trip filter threshold=0.1:** Only 7/100 pairs passed.
  Too strict for a weak SPA2MSLG model (chrF 42.84). Used threshold=0.0 instead.

---

## Resolved Bugs (non-trivial)

- **Token overflow in generation:** Fixed clipping of prediction ids before decoding
  (`np.clip(preds, 0, tokenizer.vocab_size - 1)` in `make_compute_metrics`)
- **Local model loading:** Added `local_files_only` flags in `run_evaluate.py`
  to avoid unnecessary HF hub calls when checkpoint is local

---

## Next Steps
Ordered by priority for when the test set arrives:

1. [x] **Fix bug in `run_evaluate.py`**: added missing `load_config`
2. [x] **Add forced_bos_token_id** in `generate_translations` — uses `es_XX` lang token
3. [x] **num_beams 4→5** in baseline.yaml
4. [ ] **Test on official test set** — run predict.py for both subtasks
5. [ ] **Try checkpoint ensemble** — top-3 checkpoints by chrF (free +1-3 chrF)
6. [ ] **Try special token for hyphen** — 138 occurrences, borderline sufficient
7. [ ] **LLM few-shot baseline** — paper contribution, Mistral-7B or GPT-4
8. [ ] **Add COMET to final evaluation** — already implemented, needs GPU

---

## Notes & Warnings

- T4 Colab: max batch_size=8 with LoRA r=16
- `forced_bos_token_id` MUST be set for mBART generation or output may be garbage
- `run_evaluate.py` is missing `load_config` definition — script will crash if run
- Back-translation used 100 external Spanish sentences (data/raw/external_spanish.txt)
- SPA2MSLG baseline is weak (chrF 42.84) — more back-translation pairs would help
  if SPA2MSLG is improved first
- Test files expected: `data/raw/test_mslg2spa.tsv` and `data/raw/test_spa2mslg.tsv`
