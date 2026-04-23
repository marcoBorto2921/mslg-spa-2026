# src/utils.py
"""
Shared utilities for MSLG-SPA 2026 scripts.
"""

from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_submission(
    predictions: list[str],
    output_path: Path,
    ids: list[int] | None = None,
) -> None:
    """Write predictions in the official IberLEF submission format.

    Official format (no ID):
        "SystemOutput"\\n

    Optional format (with instance ID for alignment verification):
        "InstanceIdentifier"\\t"SystemOutput"\\n

    Quotation marks are mandatory. Linux newlines required.

    Args:
        predictions:  List of system output strings.
        output_path:  Destination file path.
        ids:          Optional list of instance identifiers (same length as predictions).
    """
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        if ids is not None:
            for id_, pred in zip(ids, predictions):
                f.write(f'"{id_}"\t"{pred}"\n')
        else:
            for pred in predictions:
                f.write(f'"{pred}"\n')

    print(f"Submission saved to {output_path}")
    print(f"Lines written: {len(predictions)}")
