# MSLG-SPA 2026 — IberLEF Shared Task System

Bidirectional translation system between Mexican Sign Language (LSM) glosses
and Spanish, submitted to the
[IberLEF 2026 MSLG-SPA shared task](https://sites.google.com/view/iberlef-2026/tasks).

System description paper: `paper/mslg_spa_2026_working_notes.tex`

---

## Problem

Two complementary subtasks on a low-resource parallel corpus of 490 aligned pairs:

| Subtask | Direction | Input example | Output example |
|---|---|---|---|
| MSLG→SPA | LSM glosses → Spanish | `TÚ LLEGAR TARDE POR QUÉ` | `¿Por qué llegaste tarde?` |
| SPA→MSLG | Spanish → LSM glosses | `Vivo en América.` | `VIVIR AMERICA YO` |

LSM glosses are uppercase token sequences with structured annotations:
`dm-` (deictics), `+` (compounds), `#` (fingerspelling), `-` (multi-word units).
The task is not standard cross-lingual NMT: MSLG→SPA is morphological infilling
(reconstructing inflection absent from the gloss), while SPA→MSLG is morphological
compression (lemmatizing content words and dropping function words).

---

## Dataset

- **Source**: official competition corpus, a subset of the Lara-Ortiz LSM corpus
- **Size**: 490 aligned pairs (released to participants; full 3,000-pair corpus not available)
- **Split**: 416 train / 74 validation (85/15, seed=42)
- **Language pair**: LSM glosses ↔ Spanish
- **Avg. gloss length**: ~5 tokens; avg. Spanish length: ~8 tokens

---

## Approach

**Base model**: `facebook/mbart-large-50` (611M parameters, multilingual encoder-decoder,
pretrained on 50 languages including Spanish).

**Fine-tuning**: LoRA (rank 16, query and value projections only) via HuggingFace PEFT.
Yields 2.36M trainable parameters (0.39% of total). Full fine-tuning caused severe
overfitting from epoch 5 onward on 416 examples.

**Back-translation (BT)**: two-phase data augmentation pipeline.

- **Phase 1** (MSLG→SPA augmentation): train an initial SPA→MSLG model on real data,
  generate synthetic glosses from 344 monolingual Spanish sentences (99 manual + 245
  from the SPA→MSLG test inputs), retrain the MSLG→SPA model on 760 real+synthetic pairs.
- **Phase 2** (SPA→MSLG augmentation): use the Phase 1 MSLG→SPA model to generate
  synthetic Spanish from real glosses, yielding 832 training pairs for the reverse direction.

Quality filtering at threshold τ=0.1 kept only 7/344 pairs (too aggressive for a weak
initial model) — all 344 pairs were retained unfiltered.

---

## Results

### Validation (internal, 74 pairs, seed=42)

| Subtask | System | BLEU-4 | chrF |
|---|---|---|---|
| MSLG→SPA | Baseline (no BT) | 17.28 | 49.29 |
| MSLG→SPA | + Back-translation | **55.10** | **70.16** |
| SPA→MSLG | Baseline (no BT) | 9.70 | 44.42 |
| SPA→MSLG | + Reverse BT | **49.09** | **68.89** |

Note: validation scores for BT models are inflated by subtle leakage (74 val glosses
appear as targets in the Phase 2 synthetic training set).

### Official Test Set (blind, IberLEF 2026)

| Subtask | Submitted run | BLEU | METEOR | chrF | COMET | Task Score | Rank |
|---|---|---|---|---|---|---|---|
| MSLG→SPA | baseline (no BT) | 16.81 | 0.371 | 46.14 | 0.699 | −0.464 | 13/20 |
| SPA→MSLG | baseline + reverse BT | 14.25 | 0.361 | 47.09 | — | −0.161 | 12/19 |
| **Global** | | | | | | **−0.312** | **13/19** |

Task Score = average z-score across evaluation metrics per subtask.
COMET reported by organizers for MSLG→SPA only.

---

## Key Findings

- Back-translation provides +20 chrF on validation for both subtasks from 344 synthetic
  pairs added to a 416-pair training set — larger than typical medium-resource NMT gains.
- The BT MSLG→SPA checkpoint (70.16 val chrF) was not submitted; the baseline was submitted
  instead due to higher output variance under manual inspection. In hindsight this was
  conservative — the BT checkpoint was never evaluated on the blind test set.
- The SPA→MSLG BT model shows a large val-to-test gap (68.89 → 47.09), partially explained
  by val leakage and the small validation set (74 examples).
- LoRA r=64 with extended target modules (q,k,v,out_proj) hurt both subtasks (−3.19 chrF,
  −0.38 chrF). The compact r=16 q+v baseline was the correct choice.
- Quality-threshold BT filtering (τ=0.1) is harmful when the initial model is weak: it
  discarded 337/344 pairs, leaving a statistically useless augmented set.

---

## Repository Structure

```
mslg-spa-2026/
├── configs/
│   ├── baseline.yaml           # Final submitted config (mBART + LoRA r=16)
│   ├── baseline_bt.yaml        # BT-augmented MSLG→SPA config
│   ├── baseline_bt_s2m.yaml    # BT-augmented SPA→MSLG config
│   ├── baseline_local.yaml     # Local GPU variant (RTX 2050, batch=2)
│   └── strong.yaml             # EXP-002: LoRA r=64, rejected (overfitting)
├── data/
│   ├── raw/                    # Dataset files (not tracked — place MSLG_SPA_train.txt here)
│   └── processed/              # Cleaned data
├── final_results/              # Official evaluation PDFs from organizers
├── notebooks/
│   ├── 01_eda                  # EDA notebook
│   └── colab_training.ipynb    # End-to-end Colab training notebook
├── paper/
│   ├── mslg_spa_2026_working_notes.tex   # IberLEF 2026 working notes (LaTeX)
│   └── mslg_spa_2026.pdf                 # Compiled PDF
├── results/                    # Official submission files (.txt)
├── scripts/
│   ├── train.py                # Training entry point
│   ├── run_evaluate.py         # Evaluation entry point
│   ├── predict.py              # Generate submission files
│   ├── back_translate.py       # BT pipeline
│   ├── postprocess_spa2mslg.py # Rule-based post-processing (uppercase, stopwords)
│   └── preprocess.py           # Data validation and cleaning
├── src/
│   ├── data/                   # Dataset loading and preprocessing
│   ├── evaluation/             # BLEU, METEOR, chrF metrics
│   ├── models/                 # mBART + LoRA wrapper
│   └── utils.py
├── tests/                      # Unit tests
├── TECHNICAL_CHOICES.md        # Architecture decisions and rationale
└── requirements.txt
```

---

## Setup

> All scripts use relative paths and must be run from the project root.

```bash
git clone https://github.com/marcoBorto2921/mslg-spa-2026.git
cd mslg-spa-2026
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
pip install -e .

# Place the official training file in:
#   data/raw/MSLG_SPA_train.txt
```

### Train baseline

```bash
python scripts/train.py --config configs/baseline.yaml --subtask mslg2spa
python scripts/train.py --config configs/baseline.yaml --subtask spa2mslg
```

### Evaluate

```bash
python scripts/run_evaluate.py --config configs/baseline.yaml --subtask mslg2spa
```

### Back-translation pipeline

```bash
# Phase 1: generate synthetic glosses from Spanish pool
python scripts/back_translate.py --config configs/baseline.yaml

# Retrain on augmented data
python scripts/train.py --config configs/baseline_bt.yaml --subtask mslg2spa
python scripts/train.py --config configs/baseline_bt_s2m.yaml --subtask spa2mslg
```

### Generate submission files

```bash
python scripts/predict.py --config configs/baseline.yaml --subtask mslg2spa \
    --test_file data/raw/test_mslg2spa.txt \
    --team bortolotti --solution baseline
```

### Colab

See `notebooks/colab_training.ipynb` for the full end-to-end training pipeline
tested on a T4 GPU.

---

## Citation

```bibtex
@article{rodriguez2026overview,
  title={Overview of MSLG-SPA at IberLEF 2026: Bidirectional Translation
         between Mexican Sign Language Glosses and Spanish},
  author={Rodr{\'i}guez-Gonz{\'a}lez, Ansel Y and others},
  journal={Procesamiento del Lenguaje Natural},
  volume={77},
  year={2026}
}
```

---

## License

MIT
