# MSLG-SPA 2026

Bidirectional translation system between Mexican Sign Language (MSL) glosses and Spanish,
developed for the [MSLG-SPA 2026 shared task](https://sites.google.com/view/iberlef-2026/tasks)
at IberLEF 2026.

## Task Overview

The task consists of two complementary subtasks on a low-resource parallel corpus of
490 aligned sentence pairs:

| Subtask | Direction | Example Input | Example Output |
|---|---|---|---|
| MSLG2SPA | Glosses → Spanish | `TÚ LLEGAR TARDE POR QUÉ` | `¿Por qué llegaste tarde?` |
| SPA2MSLG | Spanish → Glosses | `Vivo en América.` | `AMÉRICA-YO VIVIR` |

MSL glosses differ fundamentally from Spanish: topic-prominent structure, no verbal
inflection, compact lexical expressions, and special annotation conventions
(hyphenated compounds, fingerspelling markers, number signs).

## Results

### MSLG2SPA (Baseline — mBART-large-50 + LoRA)

| Metric | Score |
|---|---|
| chrF | 48.09 |
| BLEU | 18.23 |
| Best epoch | 13 / 30 |

*Official evaluation results (COMET, METEOR) will be updated after May 2026.*

## Model

- **Architecture**: `facebook/mbart-large-50` (encoder-decoder transformer, 1.3B parameters)
- **Fine-tuning**: LoRA (r=16, α=32) — 2.36M trainable parameters (0.17% of total)
- **Training data**: 416 pairs (85% split), validated on 74 pairs (15% split)
- **Best checkpoint selected by**: chrF on validation set
- **Training time**: ~14 minutes on CPU / ~4 minutes on T4 GPU

## Repository Structure
```
mslg-spa-2026/
├── data/
│   ├── raw/          # Dataset files (not tracked)
│   └── processed/    # Cleaned data
├── src/
│   ├── data/         # Dataset classes and loading
│   ├── models/       # Model and LoRA wrapper
│   ├── training/     # Training utilities
│   └── evaluation/   # BLEU, chrF, METEOR, COMET, Global Score
├── scripts/
│   ├── train.py          # Training entry point
│   ├── run_evaluate.py   # Evaluation entry point
│   ├── predict.py        # Generate submission files
│   └── preprocess.py     # Data validation and cleaning
├── configs/
│   └── baseline.yaml     # Hyperparameters
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory Data Analysis
└── tests/
    └── test_dataset.py   # Unit tests (6/6 passing)
```

## Quickstart
```bash
# 1. Clone and install
git clone https://github.com/marcoBorto2921/mslg-spa-2026.git
cd mslg-spa-2026
pip install -r requirements.txt
pip install -e .

# 2. Place dataset in data/raw/MSLG_SPA_train.txt

# 3. Train
python scripts/train.py --config configs/baseline.yaml --subtask mslg2spa

# 4. Evaluate
python scripts/run_evaluate.py --config configs/baseline.yaml --subtask mslg2spa

# 5. Generate submission
python scripts/predict.py --config configs/baseline.yaml --subtask mslg2spa \
    --team YourTeam --solution baseline
```

## Requirements

Python 3.11+, PyTorch 2.x, HuggingFace Transformers 5.x, PEFT 0.10+

## Citation

If you use this code, please cite the shared task overview paper
(forthcoming, IberLEF 2026 proceedings, CEUR-WS.org).

## License

MIT