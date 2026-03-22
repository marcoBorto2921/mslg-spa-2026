# MSLG-SPA 2026

Bidirectional translation system between Mexican Sign Language (MSL) glosses and Spanish,
developed for the MSLG-SPA 2026 shared task at IberLEF 2026.

## Task Overview

Two subtasks:

| Subtask | Direction | Example Input | Example Output |
|---|---|---|---|
| MSLG2SPA | Glosses → Spanish | `TÚ LLEGAR TARDE POR QUÉ` | `¿Por qué llegaste tarde?` |
| SPA2MSLG | Spanish → Glosses | `Vivo e América.` | `AMÉRICA-YO VIVIR` |

MSL glosses differ fundamentally from Spanish: no verbal inflection, topic-prominent
structure, compact lexical expressions. This is a **low-resource** setting (489 training pairs).

## Approach

- **Baseline**: Fine-tuned `Helsinki-NLP/opus-mt-es-ROMANCE` with LoRA
- **Data augmentation**: Back-translation to expand the training set
- **Evaluation**: BLEU, METEOR, chrF (both subtasks) + COMET (MSLG2SPA only)

## Results

| Subtask | BLEU | METEOR | chrF | COMET | Global Score |
|---|---|---|---|---|---|
| MSLG2SPA | — | — | — | — | — |
| SPA2MSLG | — | — | — | — | — |

*Will be updated after official evaluation (May 2026).*

## Quickstart
```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/mslg-spa-2026.git
cd mslg-spa-2026
pip install -r requirements.txt

# 2. Place dataset files in data/raw/
#    Expected: train.tsv, test_mslg2spa.tsv, test_spa2mslg.tsv

# 3. Train
python scripts/train.py --config configs/baseline.yaml --subtask mslg2spa

# 4. Evaluate
python scripts/evaluate.py --config configs/baseline.yaml --subtask mslg2spa
```

## Requirements

Python 3.10+, PyTorch 2.x, HuggingFace Transformers 4.x

## License

MIT