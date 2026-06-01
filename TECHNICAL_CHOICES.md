# Technical Choices — MSLG-SPA 2026

Architecture decisions, rationale, and alternatives considered.

---

### Decision: Base Model
**Choice**: `facebook/mbart-large-50` (611M parameters, multilingual encoder-decoder)

**Rationale**: mBART-large-50 is pretrained on 50 languages including Spanish, giving the
model deep knowledge of Spanish morphology before any fine-tuning. Since LSM glosses do not
exist in any pretrained corpus, the value of pretraining is entirely on the Spanish side.
The encoder-decoder architecture is the natural fit for sequence-to-sequence tasks where
input and output have different lengths and structures.

**Alternatives considered**:
- `Helsinki-NLP/opus-mt-es-*`: MarianMT models trained on specific language pairs — no pair
  exists for LSM glosses, and the models are too small for low-resource generalization.
- `google/mt5-base`: viable, but requires more setup for conditional generation and offers
  no advantage over mBART on Spanish.
- Transformer from scratch: requires millions of training pairs; not feasible at 490.
- Decoder-only (GPT, LLaMA): not designed for conditional generation from a structured
  input sequence.

---

### Decision: Parameter-Efficient Fine-Tuning via LoRA
**Choice**: LoRA rank 16, query and value projections only (`q_proj`, `v_proj`), α=32,
dropout=0.1. 2.36M trainable parameters (0.39% of total).

**Rationale**: Full fine-tuning of 611M parameters on 416 examples causes severe overfitting:
validation loss began increasing from epoch 5 onward while training loss continued to decrease.
LoRA freezes all pretrained weights and injects low-rank update matrices, drastically reducing
the effective degrees of freedom. Rank 16 with q+v-only injection was the minimal configuration
that maintained training capacity without overfitting.

**Alternatives considered**:
- LoRA r=64, target_modules=[q,k,v,out_proj] (EXP-002): empirically rejected — MSLG→SPA
  chrF −3.19, SPA→MSLG chrF −0.38 vs baseline. 4.72M trainable params overfit on 490 pairs.
- Adapter layers (Houlsby 2019): more complex setup via PEFT; LoRA is better supported and
  empirically comparable.
- Prefix tuning: requires tuning a sequence of soft prompts; harder to control and less
  interpretable than LoRA rank/alpha.

---

### Decision: Back-Translation Strategy
**Choice**: Two-phase unfiltered BT. Phase 1 augments MSLG→SPA using 344 synthetic
(synth-gloss, real-Spanish) pairs. Phase 2 augments SPA→MSLG using 416 synthetic
(synth-Spanish, real-gloss) pairs.

**Rationale**: With only 416 training pairs, any additional training signal — even noisy —
substantially improves both models. The monolingual Spanish pool was extended to 344 sentences
by including the 245 Spanish sentences from the SPA→MSLG test inputs (legitimate: test inputs
are observable at submission time, no gold labels used). Phase 2 leverages the improved
Phase 1 model to generate cleaner synthetic Spanish.

**Alternatives considered**:
- Quality filtering at τ=0.1 round-trip chrF: retained 7/344 pairs — statistically useless
  when the initial SPA→MSLG model is weak (val chrF 44.42). Setting τ=0.0 retains all pairs
  and yields +20 chrF, consistent with Caswell et al. (2019).
- Iterative BT (Hoang 2018): considered but not implemented due to time constraints; multiple
  BT rounds would likely improve synthetic pair quality progressively.
- BT 4x oversample: literature cites ~4x real-data ratio as the sweet spot; not attempted
  due to limited monolingual Spanish pool.

---

### Decision: Validation Strategy
**Choice**: Single 85%/15% hold-out split (416 train / 74 val), seed=42.

**Rationale**: With 490 pairs, cross-validation folds would produce ~98-pair validation sets,
offering negligible statistical advantage over a single 74-pair split. HuggingFace Trainer
does not natively support k-fold, and implementing it would add substantial complexity.
Early stopping on val chrF with patience=10 compensates for the noisy single-split estimate.

**Alternatives considered**:
- 5-fold cross-validation: rejected for complexity and marginal benefit; Trainer API mismatch.
- Leave-one-out: too expensive and offers no reliable estimate on 490 pairs.

---

### Decision: Validation Metric for Checkpoint Selection
**Choice**: chrF (character F-score, Popović 2015).

**Rationale**: BLEU counts word-level n-gram matches and is sensitive to exact word choice —
unreliable when vocabulary is small (LSM glosses) and sequences are short (avg. 5 tokens).
chrF counts character-level n-gram matches, giving partial credit for morphological variants
(e.g., `llegaste` vs `llegó`) and handling out-of-vocabulary tokens more gracefully.

**Alternatives considered**:
- BLEU-4: rejected due to instability on short sequences with small vocabularies.
- METEOR: used as a secondary metric in final evaluation; not used for checkpoint selection
  because it requires a METEOR binary not always available in Colab.

---

### Decision: Decoder Language Token
**Choice**: Force `es_XX` (Spanish) as the decoder beginning-of-sequence token for both
subtasks.

**Rationale**: mBART uses language tokens to condition generation. LSM glosses share most
lexical roots with Spanish content words; treating them as a Spanish register reuses mBART's
Spanish generation priors and avoids language-identity mismatches. Using a different token
(e.g., `en_XX`) or none at all degrades output quality significantly.

**Alternatives considered**:
- Adding a new language token for LSM: requires vocabulary extension and additional pretraining
  to populate the embedding — not feasible with 490 pairs.
- No forced BOS: mBART generates garbage output without language conditioning.

---

### Decision: Submitted Run Selection
**Choice**: MSLG→SPA submitted the baseline (no BT); SPA→MSLG submitted the BT-augmented model.

**Rationale**: The MSLG→SPA BT checkpoint reached 70.16 val chrF but showed higher output
variance on out-of-split examples during manual inspection. The baseline (49.29 val chrF)
generalised more consistently. For SPA→MSLG, the BT model (68.89 val chrF) was the clear
best checkpoint by every metric.

**Outcome**: MSLG→SPA baseline generalised well (49.29 val → 46.14 test, −3.15 drop).
SPA→MSLG BT model showed a large val-to-test gap (68.89 → 47.09, −21.8 drop), partially
explained by val leakage: 74 val glosses appear as targets in the Phase 2 synthetic training
set.

---

### Decision: BT Data Source
**Choice**: 344 monolingual Spanish sentences (99 manually authored + 245 SPA→MSLG test inputs).

**Rationale**: The 245 test-side Spanish inputs are observable at submission time and contain
no gold labels; using them as unlabelled data is consistent with competition rules. The 99
manually authored sentences cover the same register and domain (family, school, everyday topics)
as the training corpus, improving domain alignment.

**Alternatives considered**:
- Wikipedia or news corpora: too far from the conversational register of the training corpus;
  domain mismatch would add noise.
- Larger external corpora: feasible in principle but not attempted within competition timeline.

---

## Failed Experiments

| Experiment | Change | Result | Root cause |
|---|---|---|---|
| EXP-002: LoRA r=64 + [q,k,v,out_proj] + label smoothing 0.1 | More LoRA capacity | MSLG→SPA −3.19 chrF, SPA→MSLG −0.38 chrF | Overfitting: 4.72M params on 490 pairs |
| EXP-003 Run 1: LR=5e-4, no grad clip | Tuning baseline | fp16 NaN at epoch 1.73, model corruption | Missing gradient clipping |
| EXP-003 Run 2: LR=1e-4, max_grad_norm=0.3 | Reduced LR | MSLG→SPA chrF 48.38 (−0.91 vs baseline) | LR too low; marginal regression |
| BT quality filter τ=0.1 | Filter noisy BT pairs | 7/344 pairs retained, training aborted | Weak initial model → uniformly low round-trip scores |
