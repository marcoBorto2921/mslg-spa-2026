# PROJECT_PLAN — MSLG-SPA 2026 (IberLEF)

> last_updated: 2026-04-11
> deadline system output: **2026-04-30** (19 days from today)
> deadline camera-ready paper: **2026-07-03**

Forward-looking document. Facts and decisions already made → PROJECT_LOG.md.

---

## Objective
Maximize the official IberLEF 2026 MSLG-SPA ranking metrics (**BLEU + TER + chrF**) on both
subtasks (MSLG2SPA and SPA2MSLG) before 2026-04-30, then write a working-notes paper for
CEUR-WS.org by 2026-07-03.

## Strategy
Close the gap to the reference baseline (**BARTO fine-tuned BLEU-4 = 35.0** on Lara-Ortiz 2025),
currently estimated at ~ -11 BLEU below (local best ~24). Primary levers, ranked by
expected impact:

1. **Dataset completeness.** Verify the official training size (490 vs 3000 pairs).
   6x more data likely dominates every other intervention.
2. **ASL cross-lingual transfer via ASLG-PC12.** Literature reports 62 → 85 BLEU on LSM
   with ASL data augmentation. Single highest-impact single intervention.
3. **LoRA upgrade (r=64, all-linear targets).** Codified in `configs/strong.yaml`.
4. **Back-translation oversample 4x.** Sweet spot per literature. Currently under-exploited
   (only ~100 synthetic pairs).
5. **Checkpoint ensemble** (EXP-001 already designed).
6. **Preprocessing with confirmed annotation semantics** (dm-, +, #, -).

---

## Ideas under consideration

### A. Upgrade LoRA to r=64 + all linear modules (EXP-002, code-ready)
- **Hypothesis**: LoRA capacity was the bottleneck. r=64 with q,k,v,o,fc1,fc2 unlocks
  enough capacity to absorb the gloss→text mapping without full fine-tuning overfitting.
- **Evidence**: NotebookLM C16 (LowRA framework), C17 (r=64 for <500 pairs).
- **Cost**: ~20 min Colab T4 for both subtasks. Memory may require batch_size=4 +
  gradient_accumulation=2 (34.6M trainable params → ~70M with Adam states).
- **Config**: `configs/strong.yaml` (already written).
- **Status**: code-ready, needs Colab run.

### B. Back-translation 4x oversample
- **Hypothesis**: Scaling synthetic pairs from ~100 to ~4x the real dataset (e.g. 1960 or
  12000 depending on real size) lifts BLEU substantially. NotebookLM C18 cites this as
  the exact ratio that lifted Lara-Ortiz results from 62 to 85 BLEU.
- **Dependencies**: requires a reasonable SPA2MSLG model (current chrF 42.84 is weak —
  runs noise into synthetic pairs). Best to run after EXP-002 improves SPA2MSLG.
- **Cost**: ~1-2h generation time on Colab T4 + retrain (~10 min).
- **Status**: not yet designed. Needs: (a) source of external Spanish sentences at scale
  (current `data/raw/external_spanish.txt` has 100 lines — need thousands), (b) new
  back_translate.py parameter for target count.

### C. ASL pretraining via ASLG-PC12
- **Hypothesis**: Intermediate pretraining on English-ASL provides structural priors
  (SOV order, gloss compression) transferable to Spanish-MSL.
- **Evidence**: NotebookLM C13 cites +23 BLEU on LSM (62 → 85) via ASL integration.
- **Cost**: ~2h download/prep + ~1h intermediate pretraining on Colab.
- **Risk**: language identity mismatch — ASL glosses are English-based, MSL glosses
  are Spanish-based. May need vocab alignment or simple tag-based conditioning.
- **Status**: not yet designed. Needs download of ASLG-PC12 and a new script.

### D. Preprocessing refinement with confirmed annotation semantics
- **Hypothesis**: Adding dedicated special tokens for dm- / + / # / - (instead of
  letting BPE fragment them) improves gloss comprehension.
- **Evidence**: NotebookLM B9 confirms exact semantics of each symbol.
- **Cost**: low (edit `src/data/preprocessing.py`, extend `add_hyphen_special_token`
  to a generic `add_msl_special_tokens` adding 4 tokens).
- **Risk**: special token tuning may hurt if frequencies are too low. Validate on val split.
- **Status**: not yet implemented.

### E. Iterative back-translation (Hoang 2018)
- **Hypothesis**: After BT improves SPA2MSLG, re-running BT with the improved model
  yields cleaner synthetic pairs, which further improve MSLG2SPA, etc.
- **Evidence**: NotebookLM C19 confirms valuable for <1M rows.
- **Status**: not yet designed. Candidate for a second iteration after A+B land.

### F. Curriculum learning (short→long)
- **Hypothesis**: Training short glosses first then longer ones stabilizes learning.
- **Evidence**: NotebookLM C21 cites effectiveness for low-resource MT.
- **Cost**: low-medium (custom data sampler).
- **Status**: candidate for ablation only.

### G. SALS — Semantically Aware Label Smoothing
- **Evidence**: NotebookLM C22 cites SALS as a gloss-specific label smoothing variant.
- **Status**: stretch goal — requires finding the paper and implementing custom loss.
  Plain label smoothing=0.1 is a cheaper substitute already in strong.yaml.

### H. Hybrid ensemble: Transformer + rule-based heuristics
- **Hypothesis**: For dm- / # markers, rule-based post-processing of mBART output
  could fix common failures.
- **Evidence**: NotebookLM D23 cites hybrid approaches in WSLP 2025.
- **Status**: requires error analysis first — defer until we have val-set predictions
  to inspect.

---

## Ideas rejected (with reason)

| Idea | Reason rejected |
|------|-----------------|
| METEOR/COMET in ranking pipeline | NotebookLM A1/A2 confirm official metrics are BLEU + TER + chrF only. METEOR/COMET kept as optional diagnostics. |
| Z-score composite metric | Same as above — not part of official ranking. |
| MBR decoding with chrF utility | No literature evidence for delta on this task size (NotebookLM D24). Complex to implement. Deferred. |
| LLM closed-source submission (GPT-4, Claude) | Not prohibited but no evidence of effectiveness. mBART + LoRA is the documented direction. GPT-4 only cited as data augmentation tool. |

---

## Open questions

1. **Is the 490-pair file the full official training release or a partial download?**
   Need access to the IberLEF 2026 MSLG-SPA portal to verify.
2. **Has the test set been downloaded?** Released 2026-03-27. `data/raw/` currently
   lacks `test_mslg2spa.tsv` / `test_spa2mslg.tsv`.
3. **What is the exact submission format?** NotebookLM A3 did not find details.
   Current code writes `"prediction"\n` per line — needs confirmation.
4. **Number of submissions per team?** Unknown (A4). Assume 1-3 max.
5. **Deadline timezone?** Not specified. Assume Europe/Madrid AoE-safe until confirmed.
6. **Does PEFT LoRA support fc1/fc2 on mBART correctly?** Empirically the dry-run
   instantiated fine (34.6M trainable). Training gradient flow not yet verified.

---

## Risks

1. **Dataset gap**: if the 490-pair file is the full training set, the BLEU gap to
   BARTO 35.0 is harder to close — less room for improvement. If it is partial,
   downloading the full 3000-pair version likely closes the gap immediately.
2. **T4 memory OOM on strong.yaml**: 34.6M trainable + Adam states. Mitigation ready
   (batch_size=4 + grad_accum=2).
3. **Overfitting risk from r=64 on small data**: label smoothing 0.1 + early stopping
   via `load_best_model_at_end=true` compensate partially.
4. **Test set not downloaded**: blocks final predict.py runs. Must be resolved in
   next 48h.
5. **Paper deadline July**: low urgency now but requires keeping logs of all ablations
   for the results section.

---

## Immediate next actions (Marco must do)

1. Download official test set from IberLEF portal → `data/raw/test_mslg2spa.tsv` +
   `data/raw/test_spa2mslg.tsv`.
2. Verify training set size (490 vs 3000). If larger version exists, replace.
3. Run EXP-001 (checkpoint ensemble, ~30 min Colab).
4. Run EXP-002 (strong.yaml training, ~20 min Colab). If OOM, switch to
   batch_size=4 + gradient_accumulation_steps=2.
5. Report results back for decision on next round (preprocessing refinement, BT
   oversample, ASL pretraining).
