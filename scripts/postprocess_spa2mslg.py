"""
Rule-based post-processing for SPA2MSLG model output.

MSL glosses are always UPPERCASE. The model occasionally generates lowercase
tokens or residual Spanish function words. This script normalises the output
to match the expected gloss format without modifying model weights.

Rules applied (in order):
  1. Uppercase all tokens, preserving prefix markers (dm-, #).
  2. Optionally strip common Spanish function words (--strip_stopwords flag).
     Disabled by default — risky if the model already handles them correctly.

Usage:
    # Normalise a prediction file in-place (creates .bak backup):
    python scripts/postprocess_spa2mslg.py --input outputs/Team_sol_SPA2MSLG.txt

    # Write to separate output file:
    python scripts/postprocess_spa2mslg.py \\
        --input outputs/Team_sol_SPA2MSLG.txt \\
        --output outputs/Team_sol_SPA2MSLG_pp.txt

    # Enable stopword stripping:
    python scripts/postprocess_spa2mslg.py \\
        --input outputs/Team_sol_SPA2MSLG.txt \\
        --strip_stopwords
"""

import argparse
import re
import shutil
from pathlib import Path


# Common Spanish function words that should not appear in MSL glosses.
# Only stripped when --strip_stopwords is explicitly enabled.
_SPANISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "de",
        "del",
        "en",
        "a",
        "al",
        "con",
        "por",
        "para",
        "sin",
        "que",
        "y",
        "e",
        "o",
        "u",
        "pero",
        "porque",
        "como",
        "se",
        "es",
        "son",
        "está",
        "están",
    }
)

# Gloss prefix markers that must stay lowercase
_LOWERCASE_PREFIXES: tuple[str, ...] = ("dm-", "#")


def _uppercase_token(token: str) -> str:
    """Uppercase a single gloss token, preserving prefix markers.

    Examples:
        "niño"     → "NIÑO"
        "dm-casa"  → "dm-CASA"
        "#hola"    → "#HOLA"
        "CASA"     → "CASA"   (already correct, no-op)
        "dm-CASA"  → "dm-CASA" (already correct, no-op)
    """
    for prefix in _LOWERCASE_PREFIXES:
        if token.lower().startswith(prefix):
            rest = token[len(prefix) :]
            return prefix + rest.upper()
    return token.upper()


def _uppercase_gloss_line(line: str) -> str:
    """Uppercase all tokens in a gloss line, preserving compound hyphens.

    Compound signs like LICENCIA-DE-CONDUCIR are a single token (hyphen is
    part of the gloss, not a separator). Splitting on whitespace is sufficient.
    """
    tokens = line.split()
    return " ".join(_uppercase_token(t) for t in tokens)


def _strip_stopwords_line(line: str) -> str:
    """Remove Spanish function words from a gloss line (optional, risky).

    Only removes tokens that are plain Spanish function words (no markers,
    not part of a compound). Stripped tokens are not replaced — the sequence
    simply contracts.
    """
    tokens = line.split()
    kept = [t for t in tokens if t.lower() not in _SPANISH_STOPWORDS]
    # Avoid producing an empty line for degenerate inputs
    return " ".join(kept) if kept else line


def postprocess_line(line: str, strip_stopwords: bool = False) -> str:
    """Apply all post-processing rules to a single prediction line.

    Args:
        line:             Raw model prediction (one gloss sequence).
        strip_stopwords:  If True, also strip Spanish function words.

    Returns:
        Post-processed gloss line.
    """
    line = line.strip()
    if not line:
        return line
    line = _uppercase_gloss_line(line)
    if strip_stopwords:
        line = _strip_stopwords_line(line)
    # Collapse any double spaces introduced by stripping
    line = re.sub(r" {2,}", " ", line).strip()
    return line


def postprocess_file(
    input_path: Path,
    output_path: Path,
    strip_stopwords: bool,
) -> tuple[int, int]:
    """Post-process all predictions in a file.

    Args:
        input_path:      Path to raw prediction file (one prediction per line).
        output_path:     Path to write post-processed predictions.
        strip_stopwords: Whether to strip Spanish function words.

    Returns:
        Tuple of (total_lines, lines_modified).
    """
    lines = input_path.read_text(encoding="utf-8").splitlines()

    processed: list[str] = []
    modified = 0
    for line in lines:
        pp = postprocess_line(line, strip_stopwords=strip_stopwords)
        if pp != line.strip():
            modified += 1
        processed.append(pp)

    output_path.write_text("\n".join(processed) + "\n", encoding="utf-8")
    return len(lines), modified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based post-processing for SPA2MSLG predictions."
    )
    parser.add_argument("--input", required=True, help="Input prediction file")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file path. If omitted, overwrites --input "
            "(a .bak backup is created first)."
        ),
    )
    parser.add_argument(
        "--strip_stopwords",
        action="store_true",
        help="Also strip Spanish function words (disabled by default — use with care).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output is None:
        # In-place: backup first
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup)
        print(f"Backup saved to {backup}")
        output_path = input_path
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    total, modified = postprocess_file(input_path, output_path, args.strip_stopwords)

    print("Post-processing complete.")
    print(f"  Lines total:    {total}")
    print(f"  Lines modified: {modified}  ({100 * modified / max(total, 1):.1f}%)")
    print(f"  Output:         {output_path}")
    if args.strip_stopwords:
        print("  Stopword stripping: ENABLED")


if __name__ == "__main__":
    main()
