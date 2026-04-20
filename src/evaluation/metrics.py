# src/evaluation/metrics.py
"""
Metric computation for MSLG-SPA 2026.

Official metrics (confirmed from official evaluation protocol):
  MSLG2SPA: BLEU, METEOR, chrF, COMET
  SPA2MSLG: BLEU, METEOR, chrF  (COMET not applied — gloss is not natural language)

Ranking: z-score normalization across submitted systems, then arithmetic mean.
TER is NOT part of the official ranking — kept only as an optional diagnostic.
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
        references=[[r] for r in references],  # sacrebleu expects list of lists
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
        predictions=predictions, references=[[r] for r in references]
    )
    return result["score"]


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """
    Corpus-level METEOR score. Official metric for both subtasks.

    Returns:
        METEOR score in [0, 1].
    """
    metric = evaluate.load("meteor")
    result = metric.compute(predictions=predictions, references=references)
    return result["meteor"]


def compute_comet(
    sources: list[str],
    predictions: list[str],
    references: list[str],
) -> float:
    """
    COMET score — official metric for MSLG2SPA only.
    Measures adequacy and fluency using a pretrained neural model.
    Requires the `unbabel-comet` package and GPU for reasonable speed.

    Returns:
        COMET system-level score in roughly [-1, 1], or NaN if not installed.
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
        print("[WARNING] unbabel-comet not installed. Skipping COMET.")
        return float("nan")


def compute_ter(predictions: list[str], references: list[str]) -> float:
    """
    Corpus-level TER (Translation Edit Rate) — diagnostic only, NOT official.

    Returns:
        TER score in [0, 100] — lower is better.
    """
    metric = evaluate.load("ter")
    result = metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    return result["score"]


def evaluate_subtask(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    subtask: str,
    include_comet: bool = False,
    include_ter: bool = False,
) -> dict[str, float]:
    """
    Compute official metrics for a given subtask and print results.

    Official IberLEF 2026 MSLG-SPA metrics:
      - MSLG2SPA: BLEU + METEOR + chrF + COMET (COMET opt-in via include_comet)
      - SPA2MSLG: BLEU + METEOR + chrF

    Args:
        sources:       Source sentences (required for COMET).
        predictions:   System outputs.
        references:    Gold references.
        subtask:       'mslg2spa' or 'spa2mslg'.
        include_comet: Compute COMET for mslg2spa (slow — needs unbabel-comet + GPU).
        include_ter:   Compute TER as an extra diagnostic (not in official ranking).

    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    assert subtask in ("mslg2spa", "spa2mslg")

    results: dict[str, float] = {}
    results["bleu"] = compute_bleu(predictions, references)
    results["meteor"] = compute_meteor(predictions, references)
    results["chrf"] = compute_chrf(predictions, references)

    if subtask == "mslg2spa" and include_comet:
        results["comet"] = compute_comet(sources, predictions, references)

    if include_ter:
        results["ter"] = compute_ter(predictions, references)

    # Print results table
    print(f"\n{'=' * 40}")
    print(f"  Results — {subtask.upper()}")
    print(f"{'=' * 40}")
    official = {"bleu", "meteor", "chrf", "comet"}
    for k, v in results.items():
        note = "  (diagnostic)" if k not in official else ""
        print(f"  {k.upper():<10}: {v:.4f}{note}")

    return results


def compute_global_score(
    scores_per_system: list[dict[str, float]],
    subtask: str,
) -> list[float]:
    """
    Replicate the official IberLEF 2026 Global Score for internal ablations.

    Official method: z-score normalize each metric across systems, then take
    the arithmetic mean of the normalized scores.

    Metrics used:
      - MSLG2SPA: bleu, meteor, chrf, comet (if present)
      - SPA2MSLG: bleu, meteor, chrf

    Args:
        scores_per_system: List of metric dicts, one per system.
        subtask:           'mslg2spa' or 'spa2mslg'.

    Returns:
        List of Global Scores, one per system (higher is better).
    """
    assert subtask in ("mslg2spa", "spa2mslg")

    base_metrics = ["bleu", "meteor", "chrf"]
    if subtask == "mslg2spa" and all("comet" in s for s in scores_per_system):
        base_metrics = ["bleu", "meteor", "chrf", "comet"]

    matrix = np.array(
        [[s[m] for m in base_metrics] for s in scores_per_system],
        dtype=float,
    )

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0

    normalized = (matrix - means) / stds
    global_scores = normalized.mean(axis=1).tolist()

    return global_scores
