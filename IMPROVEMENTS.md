# Possible Improvements — MSLG-SPA 2026

This document describes concrete improvements over the baseline system,
ordered by expected impact and implementation effort.

---

## 1. Back-Translation (High Impact, Medium Effort)

### What it is
Use the trained SPA2MSLG model to generate synthetic MSL glosses from
external Spanish sentences, then add these synthetic pairs to the training set
and retrain MSLG2SPA.

### Why it works
Back-translation is the standard data augmentation technique for low-resource
machine translation. It effectively doubles or triples the training data at
the cost of some noise in the synthetic glosses.

### How to implement it
```python
# 1. Collect external Spanish sentences (news, subtitles, etc.)
# 2. Run SPA2MSLG model to generate synthetic glosses
# 3. Filter by confidence score (keep only high-confidence translations)
# 4. Add to training set and retrain MSLG2SPA
```

### Expected gain
Potentially +5-10 chrF points on MSLG2SPA. The gain depends heavily on
the quality of the SPA2MSLG model and the domain of the external sentences.

---

## 2. Special Tokens for MSL Annotations (Low-Medium Impact, Low Effort)

### What it is
Add dedicated special tokens to the tokenizer vocabulary for MSL annotation
conventions, replacing the raw symbols before training.

### Annotation frequency in the training set

| Annotation | Occurrences | Percentage | Feasibility |
|---|---|---|---|
| Hyphen (-) | 138 | 28.2% | ✅ Sufficient signal |
| Fingerspelling (dm-) | 47 | 9.6% | ⚠️ Borderline |
| Compound (+) | 22 | 4.5% | ❌ Too rare |
| Number sign (#) | 5 | 1.0% | ❌ Too rare |

### Important caveat
Special tokens only help if the model sees them enough times to learn their
meaning. With 490 training pairs, adding a token that appears 5 times (#)
or 22 times (+) introduces a new vocabulary item the model cannot reliably
learn. This could hurt performance rather than help.

### Practical recommendation
- **Hyphen**: worth adding as a special token — 138 occurrences is borderline
  sufficient
- **dm-, +, #**: better handled by normalization — strip the symbol and treat
  the content as regular tokens

### Normalization alternative
Instead of special tokens, normalize the annotations:
```python
def normalize_glosses(text: str) -> str:
    text = re.sub(r'dm-', '', text)        # dm-ISABEL → ISABEL
    text = re.sub(r'\+', ' ', text)        # MAMÁ+PAPÁ → MAMÁ PAPÁ
    text = re.sub(r'#', '', text)          # #OK → OK
    # Keep hyphens — too frequent to remove blindly
    return text
```

This reduces tokenizer fragmentation without requiring the model to learn
new token meanings from scarce examples.

### Expected gain
+1-3 chrF points for hyphen special token. Normalization: unclear —
could help or hurt depending on whether the model benefits from seeing
clean tokens or loses useful structural information.

---

## 3. LLM Few-Shot Baseline (Medium Impact, Low Effort)

### What it is
Use a large language model (GPT-4, Claude, or an open model like Mistral-7B)
with few-shot prompting to translate MSL glosses, without any fine-tuning.

### Why it is interesting
With only 490 training pairs, it is not obvious that a fine-tuned small model
beats a powerful LLM given 20-30 examples in the prompt. This comparison is
a genuine research contribution — nobody has done it for MSL.

### How to implement it
```python
prompt = """Translate Mexican Sign Language glosses to Spanish.
Examples:
MSLG: TÚ LLEGAR TARDE POR QUÉ
SPA: ¿Por qué llegaste tarde?

MSLG: AMÉRICA-YO VIVIR
SPA: Vivo en América.

[... 20 more examples ...]

MSLG: {input_gloss}
SPA:"""
```

### Expected gain
Unknown — this is the interesting part. If the LLM scores comparably to
the fine-tuned model, it suggests that the translation pattern is learnable
from context alone. If the fine-tuned model wins, it confirms that even
minimal domain-specific training is valuable for specialized linguistic tasks.

---

## 4. Cross-Lingual Transfer from Other Sign Languages (High Impact, High Effort)

### What it is
Pretrain on parallel corpora from other sign languages (ASL-English,
DGS-German, LSF-French) before fine-tuning on MSL-Spanish.

### Why it works
Sign languages share structural properties — topic prominence, spatial
grammar, use of non-manual markers — regardless of the spoken language
they are paired with. A model pretrained on multiple sign language translation
tasks should develop better representations for gloss-like sequences.

### Datasets available
- ASL: How2Sign, OpenASL
- DGS: Public DGS Corpus
- LSF: MEDIAPI-SKEL

### Why it is not in the baseline
Collecting and preprocessing multiple sign language datasets is substantial
work. For a shared task baseline with a May 2026 deadline, this is a
future work contribution rather than a practical improvement.

---

## 5. Ensemble of Multiple Checkpoints (Low Impact, Low Effort)

### What it is
Instead of using only the best checkpoint, average the predictions of the
top-3 checkpoints by chrF score.

### Why it works
Ensemble methods reduce variance. Each checkpoint has memorized slightly
different aspects of the training data; averaging their predictions
produces more stable translations.

### How to implement it
Generate translations with each checkpoint separately, then use a voting
or averaging scheme over the output token probabilities.

### Expected gain
Likely +1-3 chrF points. Small but essentially free given the checkpoints
are already saved.

---

## 6. Curriculum Learning (Medium Impact, High Effort)

### What it is
Order the training examples from simple to complex — start with short,
simple gloss-Spanish pairs and gradually introduce longer, more complex
sentences with special annotations.

### Why it works
The model learns basic mapping patterns before being exposed to the harder
cases (fingerspelling, compound signs, long sentences). This can accelerate
convergence and improve final performance.

### How to implement it
Sort training pairs by gloss length or annotation complexity, then feed
them to the Trainer in order using a custom data sampler.

### Expected gain
Uncertain. Curriculum learning has mixed results in low-resource settings —
the gain depends on whether the model's bottleneck is complexity or data quantity.

---

## Summary Table

| Improvement | Impact | Effort | Recommended |
|---|---|---|---|
| Back-translation | High | Medium | ✅ Yes — implement next |
| Special tokens | Medium | Low | ✅ Yes — quick win |
| LLM few-shot baseline | Medium | Low | ✅ Yes — paper contribution |
| Cross-lingual transfer | High | High | Future work |
| Checkpoint ensemble | Low | Low | ✅ Yes — free improvement |
| Curriculum learning | Medium | High | Future work |

---

## What to do next

The most pragmatic path to a stronger system and a better paper:

1. Train SPA2MSLG baseline (same pipeline, different subtask direction)
2. Implement special tokens preprocessing
3. Run back-translation with the trained SPA2MSLG model
4. Run LLM few-shot comparison with Mistral-7B or GPT-4
5. Report all results in the system paper