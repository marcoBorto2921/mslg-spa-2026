# src/evaluation/metrics.py
"""
Metric computation for MSLG-SPA 2026.

Official metrics:
  - MSLG2SPA: BLEU, METEOR, chrF, COMET
  - SPA2MSLG: BLEU, METEOR, chrF  (no COMET — glosses are not natural language)
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
) -> dict[str, float]:
    """
    Compute all official metrics for a given subtask and print results.

    Args:
        sources:     Source sentences.
        predictions: System outputs.
        references:  Gold references.
        subtask:     'mslg2spa' or 'spa2mslg'.

    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    assert subtask in ("mslg2spa", "spa2mslg")

    results = {}
    results["bleu"]   = compute_bleu(predictions, references)
    results["chrf"]   = compute_chrf(predictions, references)
    results["meteor"] = compute_meteor(predictions, references)

    # COMET only for MSLG2SPA
    if subtask == "mslg2spa":
        results["comet"] = compute_comet(sources, predictions, references)

    # Print results table
    print(f"\n{'='*40}")
    print(f"  Results — {subtask.upper()}")
    print(f"{'='*40}")
    for k, v in results.items():
        print(f"  {k.upper():<10}: {v:.4f}")

    return results


def compute_global_score(
    scores_per_system: list[dict[str, float]],
    subtask: str
) -> list[float]:
    """
    Compute the official Global Score used for ranking.

    Each metric is z-score normalized across all systems,
    then Global Score = mean of normalized scores.

    Args:
        scores_per_system: List of metric dicts, one per system.
        subtask:           Determines which metrics to include.

    Returns:
        List of Global Scores, one per system.
    """
    metrics = ["bleu", "chrf", "meteor"]
    if subtask == "mslg2spa":
        metrics.append("comet")

    # Build matrix: shape (n_systems, n_metrics)
    matrix = np.array(
        [[s[m] for m in metrics] for s in scores_per_system],
        dtype=float
    )

    # Z-score normalize each metric column across systems
    means = matrix.mean(axis=0)
    stds  = matrix.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero

    normalized    = (matrix - means) / stds
    global_scores = normalized.mean(axis=1).tolist()

    return global_scores