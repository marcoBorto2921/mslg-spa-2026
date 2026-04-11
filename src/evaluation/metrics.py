# src/evaluation/metrics.py
"""
Metric computation for MSLG-SPA 2026.

Official metrics (confirmed via NotebookLM query, 2026-04-11):
  - BLEU, TER, chrF for both subtasks.
  - METEOR and COMET are NOT part of the official ranking — kept here as
    optional diagnostics for internal analysis only.
  - TER is lower-is-better; BLEU and chrF are higher-is-better.
"""

import evaluate
import numpy as np


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Corpus-level BLEU score.

    Args:
        predictions: List of system output strings.
        references:  List of reference strings.

    Returns:
        BLEU score in [0, 100].
    """
    metric = evaluate.load("sacrebleu")
    result = metric.compute(
        predictions=predictions,
        references=[[r] for r in references]  # sacrebleu expects list of lists
    )
    return result["score"]


def compute_ter(predictions: list[str], references: list[str]) -> float:
    """
    Corpus-level TER (Translation Edit Rate) — lower is better.

    TER measures the minimum number of edits (insertions, deletions,
    substitutions, shifts) required to transform the prediction into
    the reference, normalized by reference length. Part of the official
    IberLEF 2026 MSLG-SPA ranking (BLEU + TER + chrF).

    Returns:
        TER score in [0, 100] — 0 means perfect match.
    """
    metric = evaluate.load("ter")
    result = metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    return result["score"]


def compute_chrf(predictions: list[str], references: list[str]) -> float:
    """
    Corpus-level chrF score (character n-gram F-score).
    More robust than BLEU on short sequences and small datasets.

    Returns:
        chrF score in [0, 100].
    """
    metric = evaluate.load("chrf")
    result = metric.compute(
        predictions=predictions,
        references=[[r] for r in references]
    )
    return result["score"]


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """
    Corpus-level METEOR score.

    Returns:
        METEOR score in [0, 1].
    """
    metric = evaluate.load("meteor")
    result = metric.compute(
        predictions=predictions,
        references=references
    )
    return result["meteor"]


def compute_comet(
    sources: list[str],
    predictions: list[str],
    references: list[str],
) -> float:
    """
    COMET score — only for MSLG2SPA subtask.
    Measures adequacy and fluency using a pretrained neural model.
    Requires GPU for reasonable speed.

    Returns:
        COMET system-level score in roughly [-1, 1].
    """
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        data = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(sources, predictions, references)
        ]
        output = comet_model.predict(data, batch_size=8, gpus=0)
        return output["system_score"]
    except ImportError:
        print("[WARNING] comet not installed. Skipping COMET.")
        return float("nan")


def evaluate_subtask(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    subtask: str,
    include_diagnostics: bool = False,
) -> dict[str, float]:
    """
    Compute official metrics for a given subtask and print results.

    Official IberLEF 2026 MSLG-SPA metrics: BLEU, TER, chrF (both subtasks).

    Args:
        sources:             Source sentences.
        predictions:         System outputs.
        references:          Gold references.
        subtask:             'mslg2spa' or 'spa2mslg'.
        include_diagnostics: If True, also compute METEOR (and COMET for
                             mslg2spa). These are NOT part of the official
                             ranking — use only for internal error analysis.

    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    assert subtask in ("mslg2spa", "spa2mslg")

    results: dict[str, float] = {}
    # Official metrics
    results["bleu"] = compute_bleu(predictions, references)
    results["ter"]  = compute_ter(predictions, references)
    results["chrf"] = compute_chrf(predictions, references)

    # Optional diagnostics — NOT part of the official ranking
    if include_diagnostics:
        results["meteor"] = compute_meteor(predictions, references)
        if subtask == "mslg2spa":
            results["comet"] = compute_comet(sources, predictions, references)

    # Print results table
    print(f"\n{'='*40}")
    print(f"  Results — {subtask.upper()}")
    print(f"{'='*40}")
    for k, v in results.items():
        note = "  (diagnostic)" if k in ("meteor", "comet") else ""
        print(f"  {k.upper():<10}: {v:.4f}{note}")

    return results


def compute_global_score(
    scores_per_system: list[dict[str, float]],
    subtask: str,
) -> list[float]:
    """
    Compute a composite internal score for comparing systems.

    NOTE: The official IberLEF 2026 ranking publishes BLEU, TER, and chrF
    separately — there is no official single composite metric confirmed in
    the task description. This function combines BLEU + chrF + (1 - TER/100)
    by z-score normalization across systems as a convenience for internal
    ablations. Do NOT cite this as the official ranking metric.

    Args:
        scores_per_system: List of metric dicts, one per system.
        subtask:           Kept for API compatibility; not used in the
                           current formulation.

    Returns:
        List of composite scores, one per system (higher is better).
    """
    metrics = ["bleu", "ter", "chrf"]

    matrix = np.array(
        [[s[m] for m in metrics] for s in scores_per_system],
        dtype=float,
    )

    # Flip TER so higher = better (TER is lower-is-better)
    # Column order: bleu, ter, chrf  →  ter is column index 1
    matrix[:, 1] = -matrix[:, 1]

    means = matrix.mean(axis=0)
    stds  = matrix.std(axis=0)
    stds[stds == 0] = 1.0

    normalized    = (matrix - means) / stds
    global_scores = normalized.mean(axis=1).tolist()

    return global_scores