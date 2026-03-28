# Technical Documentation — MSLG-SPA 2026

This document describes the technical pipeline, architectural choices,
and design decisions behind the MSLG-SPA 2026 system.

---

## 1. Problem Framing

The task is sequence-to-sequence translation between two linguistically
distant representations: Mexican Sign Language (MSL) glosses and Spanish.

Key characteristics that drive every technical decision:

- **Low-resource**: 490 training pairs. This rules out training from scratch
  and makes overfitting the primary risk.
- **Asymmetric structure**: MSL glosses are topic-prominent, lack verbal
  inflection, and omit grammatical markers that must be reconstructed in Spanish.
- **Special annotations**: hyphenated compounds (`LICENCIA-DE-CONDUCIR`),
  fingerspelling (`dm-ISABEL`), compound signs (`MAMÁ+PAPÁ`), number signs (`#OK`).
- **Bidirectional**: the same training data supports both MSLG2SPA and SPA2MSLG,
  but the two directions have different evaluation protocols.

---

## 2. Model Choice: mBART-large-50

### Why an encoder-decoder architecture

Translation is a sequence-to-sequence task: the input and output are two
different sequences with different lengths and structure. This requires an
architecture with both an encoder (to understand the source) and a decoder
(to generate the target token by token).

Encoder-only models (BERT, RoBERTa) cannot generate text — they produce
representations, not sequences. Decoder-only models (GPT, LLaMA) can generate
text but are not optimized for conditional generation from a structured input.
Encoder-decoder models (T5, mBART, MarianMT) are the natural choice for translation.

### Why mBART-large-50 specifically

`facebook/mbart-large-50` was chosen over alternatives for three reasons:

1. **Spanish knowledge**: mBART was pretrained on 50 languages including Spanish,
   giving the model deep knowledge of Spanish grammar, morphology, and semantics
   before any fine-tuning. Since MSL glosses do not exist in any pretrained model,
   the Spanish side is where pretraining helps the most.

2. **Multilingual pretraining**: unlike monolingual models, mBART's multilingual
   training makes it more robust to novel token sequences like MSL glosses, which
   share surface-level character patterns with Spanish but have completely different
   grammatical structure.

3. **Encoder-decoder native**: mBART is a MarianMT-style model natively designed
   for translation, requiring no architectural adaptation.

Alternatives considered:

| Model | Reason not chosen |
|---|---|
| `Helsinki-NLP/opus-mt-es-en` | Translates Spanish → English, wrong direction |
| `google/mt5-base` | More complex to set up for translation, slower |
| Transformer from scratch | Requires millions of examples, not 490 |
| GPT/LLaMA (decoder-only) | Not designed for conditional translation |

---

## 3. Fine-tuning Strategy: LoRA

### The overfitting problem

`facebook/mbart-large-50` has 1.3 billion parameters. With 416 training pairs,
full fine-tuning would update all 1.3B parameters based on a microscopic signal.
The model would memorize the training set in a few epochs rather than learning
to generalize to new glosses.

### LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2022) addresses this by freezing all pretrained weights and
adding small trainable matrices on top of the attention layers:
```
Standard fine-tuning:  W_new = W_original + ΔW       (ΔW: millions of params)
LoRA:                  W_new = W_original + A × B     (A, B: thousands of params)
```

Where A has shape (d × r) and B has shape (r × d), with r << d.

With r=16 (rank) and α=32 (scaling), the system trains only 2.36M parameters
out of 1.38B total — 0.17% of the model. This drastically reduces overfitting
risk while preserving all pretrained Spanish knowledge.

**Configuration used:**
- `r = 16` — rank, controls capacity of the adaptation
- `lora_alpha = 32` — scaling factor (conventionally 2× r)
- `lora_dropout = 0.1` — regularization on LoRA layers
- `target_modules = ["q_proj", "v_proj"]` — applied to query and value
  projections in attention layers

---

## 4. Data Pipeline

### Loading and cleaning

The dataset is a tab-separated file with a header row. The `load_pairs`
function in `src/data/dataset.py` handles:
- Header skipping (`header=0` in pandas)
- Whitespace stripping
- Empty row removal via `dropna()` and string length filtering

### Train/validation split

With only 490 pairs, a hold-out split of 85%/15% (416 train, 74 validation)
was used instead of k-fold cross-validation. The split is seeded (`seed=42`)
for reproducibility.

K-fold was considered but rejected because the HuggingFace Trainer API does
not natively support it, and implementing it would add significant complexity
for marginal benefit on a first baseline.

### Tokenization

The HuggingFace tokenizer for mBART uses SentencePiece BPE with a vocabulary
of ~250,000 subword units. MSL glosses are tokenized as Spanish-adjacent
character sequences — `LICENCIA-DE-CONDUCIR` becomes something like
`["▁LIC", "ENCIA", "-", "DE", "-", "CON", "DUC", "IR"]`.

This is a known limitation: the tokenizer was not designed for MSL glosses and
splits them inconsistently. Special annotation markers (`dm-`, `+`, `#`) are
treated as punctuation separators rather than meaningful linguistic signals.

### Padding and label masking

All sequences are padded to `max_length=128` tokens. In the label tensor,
padding positions are replaced with `-100` — PyTorch's convention for ignoring
positions in cross-entropy loss computation. Without this, the model would be
penalized for not predicting padding tokens, which would corrupt the training signal.

---

## 5. Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `num_train_epochs` | 30 | Sufficient for convergence on 416 pairs with LoRA |
| `learning_rate` | 5e-4 | Standard for LoRA fine-tuning |
| `warmup_steps` | 50 | ~12% of first epoch, prevents early instability |
| `weight_decay` | 0.01 | L2 regularization, minor additional overfitting protection |
| `batch_size` | 8 | Fits in memory on both CPU and T4 GPU |
| `metric_for_best_model` | chrF | More stable than BLEU on small corpora |
| `num_beams` | 4 | Beam search produces better translations than greedy decoding |

### Why chrF for model selection

BLEU counts word-level n-gram matches and is sensitive to exact word choice.
chrF counts character-level n-gram matches, making it more robust when the
vocabulary is small and sequences are short — exactly our setting with MSL
glosses averaging 4.7 tokens. chrF also handles morphological variation better,
which matters for Spanish (e.g., `llegaste` vs `llegó`).

### Training results

| Epoch | chrF | BLEU | Train Loss |
|---|---|---|---|
| 1 | 16.83 | 4.63 | ~9.0 |
| 3 | 38.89 | 9.22 | ~3.0 |
| 6 | 47.19 | 14.14 | ~1.0 |
| 9 | 47.36 | 16.37 | ~0.6 |
| 13 | **48.09** | **18.23** | ~0.28 |
| 20 | 43.74 | 16.20 | ~0.08 |
| 30 | 47.73 | 18.38 | ~0.03 |

The model reached peak chrF at epoch 13 then oscillated between 45-48 for
the remaining epochs. The best checkpoint (epoch 13) is saved automatically
by `load_best_model_at_end=True`.

---

## 6. Evaluation Protocol

The official ranking uses z-score normalized metrics:

- **MSLG2SPA**: BLEU + METEOR + chrF + COMET → Global Score = mean of z-scores
- **SPA2MSLG**: BLEU + METEOR + chrF → Global Score = mean of z-scores (no COMET)

COMET is excluded from SPA2MSLG because it requires natural language output
to compute fluency scores — MSL glosses are not natural language.

The `compute_global_score` function in `src/evaluation/metrics.py` implements
this protocol exactly, allowing local simulation of the leaderboard.

---

## 7. Submission Format

Official submissions are plain text files with one translation per line,
wrapped in double quotes:
```
"SystemOutput"\n
```

The `scripts/predict.py` script generates correctly formatted files following
the naming convention `TeamName_SolutionName_SUBTASK.txt`.

## 8. Back-Translation Results

Back-translation was implemented as a data augmentation technique on the
`feature/back-translation` branch.

**Pipeline:**
1. 100 external Spanish sentences collected
2. SPA2MSLG model (chrF 42.84) used to generate synthetic MSL glosses
3. Round-trip filter tested at threshold=0.1 (kept 7/100) and 0.0 (kept 100/100)
4. Final augmented dataset: 590 pairs (490 real + 100 synthetic)
5. MSLG2SPA retrained on augmented dataset

**Result:** chrF improved from 48.09 to 52.15 (+4.06 points) on the
real validation set, confirming that back-translation is effective even
with noisy synthetic data.

---

## References

- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Liu et al. (2020). *Multilingual Denoising Pre-training for Neural Machine Translation*
  (mBART). TACL.
- Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT.