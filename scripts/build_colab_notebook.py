"""
Build notebooks/colab_training.ipynb for MSLG-SPA 2026.

Self-contained notebook that:
  1. Checks GPU, mounts Drive, installs deps
  2. Writes all source files inline via %%writefile (from current repo state)
  3. Copies training data (and optional test set) from Drive to /content/ RAM
  4. Restores any existing HuggingFace-style checkpoints from Drive -> local
  5. Trains baseline or strong config for either subtask
  6. Backs up checkpoints to Drive
  7. Generates submission files (single-model and ensemble)

Run locally: `python scripts/build_colab_notebook.py`
Output:      notebooks/colab_training.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SOURCE_FILES = [
    "src/data/dataset.py",
    "src/data/preprocessing.py",
    "src/models/seq2seq.py",
    "src/evaluation/metrics.py",
    "scripts/train.py",
    "scripts/run_evaluate.py",
    "scripts/predict.py",
    "scripts/ensemble_predict.py",
    "configs/baseline.yaml",
    "configs/strong.yaml",
]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def writefile_cell(rel_path: str) -> dict:
    """Produce a %%writefile cell for rel_path, using current repo content."""
    content = (REPO / rel_path).read_text(encoding="utf-8")
    header  = f"%%writefile /content/mslg-spa-2026/{rel_path}\n"
    return code(header + content)


def build_cells() -> list:
    cells = []

    # ---- Title --------------------------------------------------------
    cells.append(md(
        "# MSLG-SPA 2026 - Bidirectional Gloss <-> Spanish Translation\n"
        "\n"
        "**Task:** IberLEF 2026 MSLG-SPA shared task\n"
        "**Model:** mBART-large-50 + LoRA\n"
        "**Metrics (official):** BLEU + TER + chrF\n"
        "**System output deadline:** 2026-04-30\n"
        "\n"
        "---\n"
        "## Before you run\n"
        "1. `Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU`\n"
        "2. Google Drive must contain:\n"
        "   ```\n"
        "   MyDrive/ML_projects/mslg-spa-2026/data/raw/\n"
        "   |-- MSLG_SPA_train.txt          (training set, required)\n"
        "   |-- external_spanish.txt        (optional, for back-translation)\n"
        "   |-- test_mslg2spa.tsv           (required before running predict)\n"
        "   `-- test_spa2mslg.tsv           (required before running predict)\n"
        "   ```\n"
        "3. Run all cells top-to-bottom. Sections 1-4 are idempotent and safe to re-run.\n"
        "   Section 4 (checkpoint restore) is a no-op on first run and auto-resumes on subsequent runs.\n"
    ))

    # ---- Section 1: environment --------------------------------------
    cells.append(md("## 1 - Environment setup\n"))

    cells.append(code(
        "# 1.1 - GPU check\n"
        "import torch\n"
        "\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError(\n"
        "        \"No GPU detected.\\n\"\n"
        "        \"Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU\"\n"
        "    )\n"
        "\n"
        "device   = torch.device(\"cuda\")\n"
        "gpu_name = torch.cuda.get_device_name(0)\n"
        "vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
        "print(f\"GPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB\")\n"
        "print(f\"torch {torch.__version__}  |  CUDA {torch.version.cuda}\")\n"
    ))

    cells.append(code(
        "# 1.2 - Mount Drive + configure paths\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "from pathlib import Path\n"
        "\n"
        "# ============================================================\n"
        "#  CONFIGURE THESE PATHS - edit only here\n"
        "# ============================================================\n"
        "DRIVE_BASE = Path(\"/content/drive/MyDrive/ML_projects/mslg-spa-2026\")\n"
        "DRIVE_DATA = DRIVE_BASE / \"data/raw\"\n"
        "DRIVE_CKPT = DRIVE_BASE / \"checkpoints\"\n"
        "DRIVE_SUB  = DRIVE_BASE / \"submissions\"\n"
        "# ============================================================\n"
        "\n"
        "PROJECT_ROOT = Path(\"/content/mslg-spa-2026\")\n"
        "LOCAL_DATA   = Path(\"/content/data_local\")\n"
        "\n"
        "for d in [DRIVE_DATA, DRIVE_CKPT, DRIVE_SUB]:\n"
        "    d.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "print(f\"Drive base : {DRIVE_BASE}\")\n"
        "print(f\"Drive data : {DRIVE_DATA}\")\n"
        "print(f\"Drive ckpt : {DRIVE_CKPT}\")\n"
        "print(f\"Drive sub  : {DRIVE_SUB}\")\n"
    ))

    cells.append(code(
        "# 1.3 - Install packages (transformers/peft stack)\n"
        "# Colab has torch, numpy, pandas, sklearn, pyyaml already.\n"
        "!pip install -q transformers==4.46.0 peft==0.13.2 sentencepiece==0.2.0 \\\n"
        "    sacrebleu==2.4.3 evaluate==0.4.3 nltk==3.9.1\n"
        "\n"
        "import nltk\n"
        "nltk.download('punkt', quiet=True)\n"
        "nltk.download('punkt_tab', quiet=True)\n"
        "nltk.download('wordnet', quiet=True)\n"
        "print(\"Packages installed.\")\n"
    ))

    # ---- Section 2: write source files --------------------------------
    cells.append(md(
        "## 2 - Project source files\n"
        "\n"
        "Source files are written inline via `%%writefile` from the current repo state.\n"
        "To update the notebook after editing any source file, re-run `scripts/build_colab_notebook.py`.\n"
    ))

    cells.append(code(
        "# 2.0 - Directory structure\n"
        "from pathlib import Path\n"
        "\n"
        "PROJECT_ROOT = Path(\"/content/mslg-spa-2026\")\n"
        "for d in [\n"
        "    \"src/data\",\n"
        "    \"src/models\",\n"
        "    \"src/evaluation\",\n"
        "    \"src/training\",\n"
        "    \"scripts\",\n"
        "    \"configs\",\n"
        "    \"data/raw\",\n"
        "    \"data/processed\",\n"
        "    \"checkpoints\",\n"
        "    \"outputs\",\n"
        "]:\n"
        "    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "for pkg in [\"src\", \"src/data\", \"src/models\", \"src/evaluation\", \"src/training\", \"scripts\"]:\n"
        "    init = PROJECT_ROOT / pkg / \"__init__.py\"\n"
        "    if not init.exists():\n"
        "        init.write_text(\"\")\n"
        "\n"
        "print(f\"Directory structure created at {PROJECT_ROOT}\")\n"
    ))

    for src in SOURCE_FILES:
        cells.append(writefile_cell(src))

    # ---- Section 3: data setup ---------------------------------------
    cells.append(md(
        "## 3 - Data setup\n"
        "\n"
        "Copies training data from Drive into `/content/data_local/` (RAM). "
        "Test files are copied if present; otherwise the cell warns and continues.\n"
    ))

    cells.append(code(
        "# 3.1 - Copy training + test data from Drive to local RAM\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "\n"
        "LOCAL_DATA = Path(\"/content/data_local\")\n"
        "LOCAL_DATA.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "def copy_if_exists(name, required=False):\n"
        "    src = DRIVE_DATA / name\n"
        "    dst = LOCAL_DATA / name\n"
        "    if src.exists():\n"
        "        shutil.copy2(src, dst)\n"
        "        print(f\"  OK      {name}\")\n"
        "        return True\n"
        "    msg = f\"  MISSING {name}\"\n"
        "    if required:\n"
        "        raise FileNotFoundError(f\"Required file not found on Drive: {src}\")\n"
        "    print(msg + \"  (optional — skipping)\")\n"
        "    return False\n"
        "\n"
        "copy_if_exists(\"MSLG_SPA_train.txt\", required=True)\n"
        "copy_if_exists(\"external_spanish.txt\", required=False)\n"
        "has_test_m2s = copy_if_exists(\"test_mslg2spa.tsv\", required=False)\n"
        "has_test_s2m = copy_if_exists(\"test_spa2mslg.tsv\", required=False)\n"
        "\n"
        "print()\n"
        "if not (has_test_m2s and has_test_s2m):\n"
        "    print(\"WARNING: Test files missing. Training will work, prediction cells will fail.\")\n"
        "else:\n"
        "    print(\"All test files present.\")\n"
    ))

    cells.append(code(
        "# 3.2 - Patch config paths (point both baseline.yaml and strong.yaml to /content/data_local)\n"
        "import os, sys, yaml\n"
        "\n"
        "os.chdir(str(PROJECT_ROOT))\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n"
        "\n"
        "for cfg_name in [\"configs/baseline.yaml\", \"configs/strong.yaml\"]:\n"
        "    with open(cfg_name) as f:\n"
        "        cfg = yaml.safe_load(f)\n"
        "    cfg[\"data\"][\"train_file\"]    = str(LOCAL_DATA / \"MSLG_SPA_train.txt\")\n"
        "    cfg[\"data\"][\"test_mslg2spa\"] = str(LOCAL_DATA / \"test_mslg2spa.tsv\")\n"
        "    cfg[\"data\"][\"test_spa2mslg\"] = str(LOCAL_DATA / \"test_spa2mslg.tsv\")\n"
        "    with open(cfg_name, \"w\") as f:\n"
        "        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)\n"
        "    print(f\"Patched {cfg_name}\")\n"
        "\n"
        "# Quick sanity check on the training file\n"
        "from src.data.dataset import load_pairs, print_stats\n"
        "df = load_pairs(str(LOCAL_DATA / \"MSLG_SPA_train.txt\"))\n"
        "print_stats(df, name=\"Training data\")\n"
    ))

    # ---- Section 4: checkpoint restore --------------------------------
    cells.append(md(
        "## 4 - Restore checkpoints from Drive\n"
        "\n"
        "Always run this cell before training. It is a no-op on first run and on subsequent runs "
        "it mirrors Drive's checkpoint directories into local. Completed subtasks are skipped "
        "automatically by the training cells.\n"
    ))

    cells.append(code(
        "# 4.0 - Restore any existing checkpoints from Drive\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "\n"
        "LOCAL_CKPT_ROOT = PROJECT_ROOT / \"checkpoints\"\n"
        "LOCAL_CKPT_ROOT.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "copied = 0\n"
        "if DRIVE_CKPT.exists():\n"
        "    for item in DRIVE_CKPT.iterdir():\n"
        "        target = LOCAL_CKPT_ROOT / item.name\n"
        "        if target.exists():\n"
        "            continue\n"
        "        if item.is_dir():\n"
        "            shutil.copytree(item, target)\n"
        "        else:\n"
        "            shutil.copy2(item, target)\n"
        "        copied += 1\n"
        "        print(f\"  restored  {item.name}\")\n"
        "\n"
        "if copied == 0:\n"
        "    print(\"No checkpoints on Drive. Fresh training run.\")\n"
        "else:\n"
        "    print(f\"Restored {copied} item(s) from Drive.\")\n"
    ))

    # ---- Section 5: training -----------------------------------------
    cells.append(md(
        "## 5 - Training\n"
        "\n"
        "Pick a config (`baseline.yaml` for the reference chrF 52.15 run, or `strong.yaml` for the "
        "LoRA r=64 + all-linear-targets upgrade) and run both subtasks.\n"
        "\n"
        "**Memory note** — `strong.yaml` has 34.6M trainable params. If you hit OOM on T4, lower "
        "`per_device_train_batch_size` in the config cell below to 4.\n"
    ))

    cells.append(code(
        "# 5.1 - Choose config\n"
        "# ============================================================\n"
        "#  EDIT HERE\n"
        "# ============================================================\n"
        "CONFIG_NAME = \"baseline.yaml\"  # or \"strong.yaml\"\n"
        "# ============================================================\n"
        "\n"
        "CONFIG_PATH = f\"configs/{CONFIG_NAME}\"\n"
        "import yaml\n"
        "with open(CONFIG_PATH) as f:\n"
        "    cfg = yaml.safe_load(f)\n"
        "print(f\"Config        : {CONFIG_PATH}\")\n"
        "print(f\"Model         : {cfg['model']['name']}\")\n"
        "print(f\"LoRA r        : {cfg['lora']['r']}\")\n"
        "print(f\"LoRA targets  : {cfg['lora'].get('target_modules', '[q_proj,v_proj]')}\")\n"
        "print(f\"Epochs        : {cfg['training']['num_train_epochs']}\")\n"
        "print(f\"Batch         : {cfg['training']['per_device_train_batch_size']}\")\n"
        "print(f\"Label smooth  : {cfg['training'].get('label_smoothing_factor', 0.0)}\")\n"
    ))

    cells.append(code(
        "# 5.2 - Train MSLG2SPA\n"
        "# Output dir from config is relative, so checkpoints land in /content/mslg-spa-2026/<output_dir>\n"
        "import os\n"
        "os.chdir(str(PROJECT_ROOT))\n"
        "\n"
        "!python scripts/train.py --config {CONFIG_PATH} --subtask mslg2spa\n"
    ))

    cells.append(code(
        "# 5.3 - Train SPA2MSLG\n"
        "import os\n"
        "os.chdir(str(PROJECT_ROOT))\n"
        "\n"
        "!python scripts/train.py --config {CONFIG_PATH} --subtask spa2mslg\n"
    ))

    cells.append(code(
        "# 5.4 - Backup all checkpoints to Drive\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "\n"
        "LOCAL_CKPT_ROOT = PROJECT_ROOT / \"checkpoints\"\n"
        "count = 0\n"
        "for item in LOCAL_CKPT_ROOT.iterdir():\n"
        "    target = DRIVE_CKPT / item.name\n"
        "    if item.is_dir():\n"
        "        if target.exists():\n"
        "            shutil.rmtree(target)\n"
        "        shutil.copytree(item, target)\n"
        "    else:\n"
        "        shutil.copy2(item, target)\n"
        "    count += 1\n"
        "    print(f\"  backed up  {item.name}\")\n"
        "\n"
        "print(f\"\\nBacked up {count} item(s) to {DRIVE_CKPT}\")\n"
    ))

    # ---- Section 6: evaluate / predict --------------------------------
    cells.append(md(
        "## 6 - Evaluation and submission\n"
        "\n"
        "Runs `run_evaluate.py` if test files have references, or `predict.py` / `ensemble_predict.py` "
        "for blind submission.\n"
    ))

    cells.append(code(
        "# 6.1 - Submission parameters\n"
        "# ============================================================\n"
        "#  EDIT HERE\n"
        "# ============================================================\n"
        "TEAM_NAME     = \"mslgTeam\"         # your team name\n"
        "SOLUTION_NAME = \"baseline_bt\"       # label for this submission\n"
        "N_CHECKPOINTS = 3                    # for ensemble_predict\n"
        "# ============================================================\n"
        "print(f\"Team     : {TEAM_NAME}\")\n"
        "print(f\"Solution : {SOLUTION_NAME}\")\n"
    ))

    cells.append(code(
        "# 6.2 - Single-model submission (MSLG2SPA)\n"
        "import os\n"
        "os.chdir(str(PROJECT_ROOT))\n"
        "!python scripts/predict.py --config {CONFIG_PATH} --subtask mslg2spa \\\n"
        "    --team {TEAM_NAME} --solution {SOLUTION_NAME}\n"
    ))

    cells.append(code(
        "# 6.3 - Single-model submission (SPA2MSLG)\n"
        "import os\n"
        "os.chdir(str(PROJECT_ROOT))\n"
        "!python scripts/predict.py --config {CONFIG_PATH} --subtask spa2mslg \\\n"
        "    --team {TEAM_NAME} --solution {SOLUTION_NAME}\n"
    ))

    cells.append(code(
        "# 6.4 - Ensemble submission (top-N checkpoints)\n"
        "# Point --checkpoint_dir at the subtask-specific folder. Baseline writes to\n"
        "# checkpoints/baseline and strong writes to checkpoints/strong — both are shared\n"
        "# between subtasks unless the config was split. If a per-subtask subfolder exists,\n"
        "# prefer that one; otherwise use the config output_dir.\n"
        "import os, yaml\n"
        "os.chdir(str(PROJECT_ROOT))\n"
        "\n"
        "with open(CONFIG_PATH) as f:\n"
        "    _cfg = yaml.safe_load(f)\n"
        "CKPT_DIR = _cfg[\"training\"][\"output_dir\"]\n"
        "print(f\"Checkpoint dir: {CKPT_DIR}\")\n"
        "\n"
        "!python scripts/ensemble_predict.py --config {CONFIG_PATH} --subtask mslg2spa \\\n"
        "    --checkpoint_dir {CKPT_DIR} --team {TEAM_NAME} --solution {SOLUTION_NAME}_ensemble{N_CHECKPOINTS} \\\n"
        "    --n_checkpoints {N_CHECKPOINTS}\n"
        "!python scripts/ensemble_predict.py --config {CONFIG_PATH} --subtask spa2mslg \\\n"
        "    --checkpoint_dir {CKPT_DIR} --team {TEAM_NAME} --solution {SOLUTION_NAME}_ensemble{N_CHECKPOINTS} \\\n"
        "    --n_checkpoints {N_CHECKPOINTS}\n"
    ))

    cells.append(code(
        "# 6.5 - Evaluate on val split (no test references needed) - ensemble vs single-best\n"
        "# Optional sanity check before submission. Uses the same val split as training.\n"
        "import os, yaml\n"
        "os.chdir(str(PROJECT_ROOT))\n"
        "\n"
        "with open(CONFIG_PATH) as f:\n"
        "    _cfg = yaml.safe_load(f)\n"
        "CKPT_DIR = _cfg[\"training\"][\"output_dir\"]\n"
        "\n"
        "!python scripts/ensemble_predict.py --config {CONFIG_PATH} --subtask mslg2spa \\\n"
        "    --checkpoint_dir {CKPT_DIR} --validate --n_checkpoints {N_CHECKPOINTS}\n"
        "!python scripts/ensemble_predict.py --config {CONFIG_PATH} --subtask spa2mslg \\\n"
        "    --checkpoint_dir {CKPT_DIR} --validate --n_checkpoints {N_CHECKPOINTS}\n"
    ))

    cells.append(code(
        "# 6.6 - Copy all outputs to Drive + download locally\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "from google.colab import files\n"
        "\n"
        "outputs_dir = PROJECT_ROOT / \"outputs\"\n"
        "if not outputs_dir.exists() or not any(outputs_dir.iterdir()):\n"
        "    print(\"No outputs to upload.\")\n"
        "else:\n"
        "    for f in outputs_dir.glob(\"*.txt\"):\n"
        "        dest = DRIVE_SUB / f.name\n"
        "        shutil.copy2(f, dest)\n"
        "        print(f\"  copied to Drive: {dest}\")\n"
        "        files.download(str(f))\n"
    ))

    return cells


def build_notebook() -> dict:
    return {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = REPO / "notebooks" / "colab_training.ipynb"
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_cells = len(nb["cells"])
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.relative_to(REPO)}  ({n_cells} cells, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
