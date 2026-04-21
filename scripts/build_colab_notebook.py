"""
Build notebooks/colab_training.ipynb for MSLG-SPA 2026.

Notebook structure:
  1. Checks GPU, mounts Drive, installs deps
  2. git clone / git pull from GitHub (no inline %%writefile)
  3. Copies training data from Drive to /content/ RAM
  4. Restores checkpoints from Drive -> local
  5. Training
  6. Submission generation (baseline, baseline_bt, ensemble)

Run locally: `python scripts/build_colab_notebook.py`
Output:      notebooks/colab_training.ipynb
"""

from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


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
    header = f"%%writefile /content/mslg-spa-2026/{rel_path}\n"
    return code(header + content)


def build_cells() -> list:
    cells = []

    # ---- Title --------------------------------------------------------
    cells.append(
        md(
            "# MSLG-SPA 2026 - Bidirectional Gloss <-> Spanish Translation\n"
            "\n"
            "**Task:** IberLEF 2026 MSLG-SPA shared task\n"
            "**Model:** mBART-large-50 + LoRA\n"
            "**Metrics (official):** BLEU + METEOR + chrF (+ COMET for MSLG2SPA)\n"
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
        )
    )

    # ---- Section 1: environment --------------------------------------
    cells.append(md("## 1 - Environment setup\n"))

    cells.append(
        code(
            "# 1.1 - GPU check\n"
            "import torch\n"
            "\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError(\n"
            '        "No GPU detected.\\n"\n'
            '        "Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU"\n'
            "    )\n"
            "\n"
            'device   = torch.device("cuda")\n'
            "gpu_name = torch.cuda.get_device_name(0)\n"
            "vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
            'print(f"GPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")\n'
            'print(f"torch {torch.__version__}  |  CUDA {torch.version.cuda}")\n'
        )
    )

    cells.append(
        code(
            "# 1.2 - Mount Drive + configure paths\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "\n"
            "from pathlib import Path\n"
            "\n"
            "# ============================================================\n"
            "#  CONFIGURE THESE PATHS - edit only here\n"
            "# ============================================================\n"
            'DRIVE_BASE = Path("/content/drive/MyDrive/ML_projects/mslg-spa-2026")\n'
            'DRIVE_DATA = DRIVE_BASE / "data/raw"\n'
            'DRIVE_CKPT = DRIVE_BASE / "checkpoints"\n'
            'DRIVE_SUB  = DRIVE_BASE / "submissions"\n'
            "# ============================================================\n"
            "\n"
            'PROJECT_ROOT = Path("/content/mslg-spa-2026")\n'
            'LOCAL_DATA   = Path("/content/data_local")\n'
            "\n"
            "for d in [DRIVE_DATA, DRIVE_CKPT, DRIVE_SUB]:\n"
            "    d.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            'print(f"Drive base : {DRIVE_BASE}")\n'
            'print(f"Drive data : {DRIVE_DATA}")\n'
            'print(f"Drive ckpt : {DRIVE_CKPT}")\n'
            'print(f"Drive sub  : {DRIVE_SUB}")\n'
        )
    )

    cells.append(
        code(
            "# 1.3 - Install packages (transformers/peft stack)\n"
            "# Colab has torch, numpy, pandas, sklearn, pyyaml already.\n"
            "# unbabel-comet omitted: COMET is computed by organizers server-side;\n"
            "# local z-score would be meaningless without other teams' scores.\n"
            "!pip install -q transformers==4.46.0 peft==0.13.2 sentencepiece==0.2.0 \\\n"
            "    sacrebleu==2.4.3 evaluate==0.4.3 nltk==3.9.1\n"
            "\n"
            "import nltk\n"
            "nltk.download('punkt', quiet=True)\n"
            "nltk.download('punkt_tab', quiet=True)\n"
            "nltk.download('wordnet', quiet=True)\n"
            'print("Packages installed.")\n'
        )
    )

    # ---- Section 2: clone / pull repo ---------------------------------
    cells.append(
        md(
            "## 2 - Clone / update repo\n"
            "\n"
            "Clones the repo on first run, pulls the latest code on subsequent runs.\n"
            "All source files come from GitHub — no inline `%%writefile` needed.\n"
        )
    )

    cells.append(
        code(
            "# 2.1 - Clone or pull repo\n"
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            'PROJECT_ROOT = Path("/content/mslg-spa-2026")\n'
            "\n"
            "if PROJECT_ROOT.exists():\n"
            "    print('Repo exists — pulling latest...')\n"
            "    !git -C {PROJECT_ROOT} pull\n"
            "else:\n"
            "    print('Cloning repo...')\n"
            "    !git clone https://github.com/marcoBorto2921/mslg-spa-2026.git {PROJECT_ROOT}\n"
            "\n"
            "# Create dirs not tracked by git (empty dirs)\n"
            'for d in ["data/raw", "data/processed", "checkpoints", "outputs"]:\n'
            "    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "import sys\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            'print(f"Project root: {PROJECT_ROOT}")\n'
        )
    )

    # ---- Section 3: data setup ---------------------------------------
    cells.append(
        md(
            "## 3 - Data setup\n"
            "\n"
            "Copies training data from Drive into `/content/data_local/` (RAM). "
            "Test files are copied if present; otherwise the cell warns and continues.\n"
        )
    )

    cells.append(
        code(
            "# 3.1 - Copy training + test data from Drive to local RAM\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "\n"
            'LOCAL_DATA = Path("/content/data_local")\n'
            "LOCAL_DATA.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "def copy_if_exists(name, required=False):\n"
            "    src = DRIVE_DATA / name\n"
            "    dst = LOCAL_DATA / name\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, dst)\n"
            '        print(f"  OK      {name}")\n'
            "        return True\n"
            '    msg = f"  MISSING {name}"\n'
            "    if required:\n"
            '        raise FileNotFoundError(f"Required file not found on Drive: {src}")\n'
            '    print(msg + "  (optional — skipping)")\n'
            "    return False\n"
            "\n"
            'copy_if_exists("MSLG_SPA_train.txt", required=True)\n'
            'copy_if_exists("external_spanish.txt", required=False)\n'
            'copy_if_exists("augmented_train.tsv", required=False)  # forward BT (SPA→MSLG synthetic)\n'
            'copy_if_exists("augmented_train_reverse.tsv", required=False)  # reverse BT (MSLG→SPA synthetic)\n'
            'has_test_m2s = copy_if_exists("test_mslg2spa.tsv", required=False)\n'
            'has_test_s2m = copy_if_exists("test_spa2mslg.tsv", required=False)\n'
            "\n"
            "print()\n"
            "if not (has_test_m2s and has_test_s2m):\n"
            '    print("WARNING: Test files missing. Training will work, prediction cells will fail.")\n'
            "else:\n"
            '    print("All test files present.")\n'
        )
    )

    cells.append(
        code(
            "# 3.2 - Patch config paths (point both baseline.yaml and strong.yaml to /content/data_local)\n"
            "import os, sys, yaml\n"
            "\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            'for cfg_name in ["configs/baseline.yaml", "configs/strong.yaml"]:\n'
            "    with open(cfg_name) as f:\n"
            "        cfg = yaml.safe_load(f)\n"
            '    cfg["data"]["train_file"]    = str(LOCAL_DATA / "MSLG_SPA_train.txt")\n'
            '    cfg["data"]["test_mslg2spa"] = str(LOCAL_DATA / "test_mslg2spa.tsv")\n'
            '    cfg["data"]["test_spa2mslg"] = str(LOCAL_DATA / "test_spa2mslg.tsv")\n'
            '    with open(cfg_name, "w") as f:\n'
            "        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)\n"
            '    print(f"Patched {cfg_name}")\n'
            "\n"
            "# baseline_bt.yaml: augmented_train.tsv (SPA→MSLG synthetic, for MSLG2SPA)\n"
            '_bt_cfg_name = "configs/baseline_bt.yaml"\n'
            "with open(_bt_cfg_name) as f:\n"
            "    _bt_cfg = yaml.safe_load(f)\n"
            '_bt_cfg["data"]["real_train_file"] = str(LOCAL_DATA / "MSLG_SPA_train.txt")\n'
            '_bt_cfg["data"]["train_file"]      = str(LOCAL_DATA / "augmented_train.tsv")\n'
            '_bt_cfg["data"]["test_mslg2spa"]   = str(LOCAL_DATA / "test_mslg2spa.tsv")\n'
            '_bt_cfg["data"]["test_spa2mslg"]   = str(LOCAL_DATA / "test_spa2mslg.tsv")\n'
            'with open(_bt_cfg_name, "w") as f:\n'
            "    yaml.dump(_bt_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)\n"
            'print(f"Patched {_bt_cfg_name}")\n'
            "\n"
            "# baseline_bt_s2m.yaml: augmented_train_reverse.tsv (MSLG→SPA synthetic, for SPA2MSLG)\n"
            '_s2m_cfg_name = "configs/baseline_bt_s2m.yaml"\n'
            "with open(_s2m_cfg_name) as f:\n"
            "    _s2m_cfg = yaml.safe_load(f)\n"
            '_s2m_cfg["data"]["real_train_file"] = str(LOCAL_DATA / "MSLG_SPA_train.txt")\n'
            '_s2m_cfg["data"]["train_file"]      = str(LOCAL_DATA / "augmented_train_reverse.tsv")\n'
            '_s2m_cfg["data"]["test_mslg2spa"]   = str(LOCAL_DATA / "test_mslg2spa.tsv")\n'
            '_s2m_cfg["data"]["test_spa2mslg"]   = str(LOCAL_DATA / "test_spa2mslg.tsv")\n'
            'with open(_s2m_cfg_name, "w") as f:\n'
            "    yaml.dump(_s2m_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)\n"
            'print(f"Patched {_s2m_cfg_name}")\n'
            "\n"
            "# Quick sanity check on the training file\n"
            "from src.data.dataset import load_pairs, print_stats\n"
            'df = load_pairs(str(LOCAL_DATA / "MSLG_SPA_train.txt"))\n'
            'print_stats(df, name="Training data")\n'
        )
    )

    # ---- Section 4: checkpoint restore --------------------------------
    cells.append(
        md(
            "## 4 - Restore checkpoints from Drive\n"
            "\n"
            "Always run this cell before training. It is a no-op on first run and on subsequent runs "
            "it mirrors Drive's checkpoint directories into local. Completed subtasks are skipped "
            "automatically by the training cells.\n"
        )
    )

    cells.append(
        code(
            "# 4.0 - Restore any existing checkpoints from Drive\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "\n"
            'LOCAL_CKPT_ROOT = PROJECT_ROOT / "checkpoints"\n'
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
            '        print(f"  restored  {item.name}")\n'
            "\n"
            "if copied == 0:\n"
            '    print("No checkpoints on Drive. Fresh training run.")\n'
            "else:\n"
            '    print(f"Restored {copied} item(s) from Drive.")\n'
        )
    )

    # ---- Section 5: training -----------------------------------------
    cells.append(
        md(
            "## 5 - Training\n"
            "\n"
            "Two separate training blocks — **5A** for the baseline (no back-translation) "
            "and **5B** for the BT-augmented model. Run 5A first, then generate BT data (5.3), "
            "then run 5B. Each block saves best checkpoints to Drive automatically.\n"
        )
    )

    # --- 5A: baseline (no BT) ---
    cells.append(md("### 5A — Baseline (no back-translation) · `baseline.yaml`\n"))

    cells.append(
        code(
            "# 5.1 - Train MSLG2SPA — baseline (no BT)\n"
            "import os\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "\n"
            "!python scripts/train.py --config configs/baseline.yaml --subtask mslg2spa \\\n"
            "    --drive_ckpt_dir {DRIVE_CKPT}\n"
        )
    )

    cells.append(
        code(
            "# 5.2 - Train SPA2MSLG — baseline (no BT)\n"
            "import os\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "\n"
            "!python scripts/train.py --config configs/baseline.yaml --subtask spa2mslg \\\n"
            "    --drive_ckpt_dir {DRIVE_CKPT}\n"
        )
    )

    # --- BT data generation ---
    cells.append(
        code(
            "# 5.3 - Generate back-translation data (requires baseline checkpoints from 5.1/5.2)\n"
            "# Skip automatically if augmented_train.tsv already exists on Drive.\n"
            "import os, shutil\n"
            "from pathlib import Path\n"
            "\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            'aug_dst = LOCAL_DATA / "augmented_train.tsv"\n'
            "if aug_dst.exists():\n"
            '    print(f"augmented_train.tsv already exists — skipping generation.")\n'
            "else:\n"
            '    print("Generating back-translation data...")\n'
            "    !python scripts/back_translate.py \\\n"
            "        --config configs/baseline.yaml \\\n"
            "        --spa2mslg_checkpoint {PROJECT_ROOT}/checkpoints/baseline/spa2mslg/final \\\n"
            "        --mslg2spa_checkpoint {PROJECT_ROOT}/checkpoints/baseline/mslg2spa/final \\\n"
            "        --extract_from_train \\\n"
            "        --spa_file {LOCAL_DATA}/external_spanish.txt \\\n"
            "        --output {aug_dst} \\\n"
            "        --round_trip_threshold 0.0\n"
            "    if aug_dst.exists():\n"
            "        shutil.copy2(aug_dst, DRIVE_DATA / aug_dst.name)\n"
            '        print("Backed up augmented_train.tsv to Drive.")\n'
            "\n"
            "# Reverse BT: MSLG→SPA synthetic pairs — helps SPA2MSLG training\n"
            "# Uses the stronger baseline_bt MSLG2SPA model (train 5.4 first)\n"
            'rev_dst = LOCAL_DATA / "augmented_train_reverse.tsv"\n'
            "if rev_dst.exists():\n"
            '    print("augmented_train_reverse.tsv already exists — skipping generation.")\n'
            "else:\n"
            '    print("Generating reverse back-translation data (MSLG→SPA)...")\n'
            "    !python scripts/back_translate.py \\\n"
            "        --config configs/baseline.yaml \\\n"
            "        --mslg2spa_checkpoint {PROJECT_ROOT}/checkpoints/baseline_bt/mslg2spa/final \\\n"
            "        --spa2mslg_checkpoint {PROJECT_ROOT}/checkpoints/baseline/spa2mslg/final \\\n"
            "        --output {rev_dst} \\\n"
            "        --direction mslg2spa \\\n"
            "        --round_trip_threshold 0.0\n"
            "    if rev_dst.exists():\n"
            "        shutil.copy2(rev_dst, DRIVE_DATA / rev_dst.name)\n"
            '        print("Backed up augmented_train_reverse.tsv to Drive.")\n'
        )
    )

    # --- 5B: baseline + BT ---
    cells.append(md("### 5B — Baseline + back-translation · `baseline_bt.yaml`\n"))

    cells.append(
        code(
            "# 5.4 - Train MSLG2SPA — baseline + BT\n"
            "import os\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "\n"
            "!python scripts/train.py --config configs/baseline_bt.yaml --subtask mslg2spa \\\n"
            "    --drive_ckpt_dir {DRIVE_CKPT}\n"
        )
    )

    cells.append(
        code(
            "# 5.5 - Train SPA2MSLG — baseline + reverse BT (MSLG→SPA synthetic)\n"
            "import os\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "\n"
            "!python scripts/train.py --config configs/baseline_bt_s2m.yaml --subtask spa2mslg \\\n"
            "    --drive_ckpt_dir {DRIVE_CKPT}\n"
        )
    )

    cells.append(
        code(
            "# 5.6 - Backup all checkpoints to Drive\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "\n"
            'LOCAL_CKPT_ROOT = PROJECT_ROOT / "checkpoints"\n'
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
            '    print(f"  backed up  {item.name}")\n'
            "\n"
            'print(f"\\nBacked up {count} item(s) to {DRIVE_CKPT}")\n'
        )
    )

    # ---- Section 6: evaluate / predict --------------------------------
    cells.append(
        md(
            "## 6 - Submission\n"
            "\n"
            "Generates **three submission sets** for both subtasks:\n"
            "- **A** `baseline` — single-best model trained without back-translation\n"
            "- **B** `baseline_bt` — single-best model trained with back-translated data\n"
            "- **C** `baseline_bt_ensemble3` — top-3 checkpoint ensemble from the BT model\n"
            "\n"
            "Run cell 6.1 first (sets params), then 6.2–6.4 in any order. "
            "Cell 6.5 is an optional val sanity check (no test labels needed). "
            "Cell 6.6 copies everything to Drive.\n"
        )
    )

    cells.append(
        code(
            "# 6.1 - Submission parameters\n"
            "# ============================================================\n"
            "#  EDIT HERE\n"
            "# ============================================================\n"
            'TEAM_NAME     = "mslgTeam"   # your team name\n'
            "N_CHECKPOINTS = 3             # top-N for ensemble\n"
            "# ============================================================\n"
            "import os, yaml\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            'print(f"Team          : {TEAM_NAME}")\n'
            'print(f"N checkpoints : {N_CHECKPOINTS}")\n'
        )
    )

    cells.append(
        code(
            "# 6.2 - Submission A: baseline (no BT) — single-best model\n"
            "# Outputs: {TEAM_NAME}_baseline_MSLG2SPA.txt\n"
            "#          {TEAM_NAME}_baseline_SPA2MSLG.txt\n"
            "import os\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "!python scripts/predict.py --config configs/baseline.yaml \\\n"
            "    --subtask mslg2spa --team {TEAM_NAME} --solution baseline\n"
            "!python scripts/predict.py --config configs/baseline.yaml \\\n"
            "    --subtask spa2mslg --team {TEAM_NAME} --solution baseline\n"
        )
    )

    cells.append(
        code(
            "# 6.3 - Submission B: baseline_bt (with BT) — single-best model\n"
            "# Outputs: {TEAM_NAME}_baseline_bt_MSLG2SPA.txt\n"
            "#          {TEAM_NAME}_baseline_bt_SPA2MSLG.txt\n"
            "import os\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "!python scripts/predict.py --config configs/baseline_bt.yaml \\\n"
            "    --subtask mslg2spa --team {TEAM_NAME} --solution baseline_bt\n"
            "!python scripts/predict.py --config configs/baseline_bt.yaml \\\n"
            "    --subtask spa2mslg --team {TEAM_NAME} --solution baseline_bt\n"
        )
    )

    cells.append(
        code(
            "# 6.4 - Submission C: baseline_bt top-3 ensemble\n"
            "# Outputs: {TEAM_NAME}_baseline_bt_ensemble3_MSLG2SPA.txt\n"
            "#          {TEAM_NAME}_baseline_bt_ensemble3_SPA2MSLG.txt\n"
            "import os, yaml\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "with open('configs/baseline_bt.yaml') as f:\n"
            "    _bt = yaml.safe_load(f)\n"
            "_bt_base = _bt['training']['output_dir']\n"
            'print(f"Checkpoint base: {_bt_base}")\n'
            "\n"
            "!python scripts/ensemble_predict.py --config configs/baseline_bt.yaml \\\n"
            "    --subtask mslg2spa \\\n"
            "    --checkpoint_dir {_bt_base}/mslg2spa \\\n"
            "    --team {TEAM_NAME} --solution baseline_bt_ensemble{N_CHECKPOINTS} \\\n"
            "    --n_checkpoints {N_CHECKPOINTS}\n"
            "!python scripts/ensemble_predict.py --config configs/baseline_bt.yaml \\\n"
            "    --subtask spa2mslg \\\n"
            "    --checkpoint_dir {_bt_base}/spa2mslg \\\n"
            "    --team {TEAM_NAME} --solution baseline_bt_ensemble{N_CHECKPOINTS} \\\n"
            "    --n_checkpoints {N_CHECKPOINTS}\n"
        )
    )

    cells.append(
        code(
            "# 6.5 - Val sanity check: single-best vs ensemble on real val split\n"
            "# No test labels needed. Compares chrF of single-best vs top-3 ensemble.\n"
            "# Run this before deciding which submission to use.\n"
            "import os, yaml\n"
            "os.chdir(str(PROJECT_ROOT))\n"
            "with open('configs/baseline_bt.yaml') as f:\n"
            "    _bt = yaml.safe_load(f)\n"
            "_bt_base = _bt['training']['output_dir']\n"
            "\n"
            'print("=== MSLG2SPA ===")\n'
            "!python scripts/ensemble_predict.py --config configs/baseline_bt.yaml \\\n"
            "    --subtask mslg2spa \\\n"
            "    --checkpoint_dir {_bt_base}/mslg2spa \\\n"
            "    --validate --n_checkpoints {N_CHECKPOINTS}\n"
            'print("=== SPA2MSLG ===")\n'
            "!python scripts/ensemble_predict.py --config configs/baseline_bt.yaml \\\n"
            "    --subtask spa2mslg \\\n"
            "    --checkpoint_dir {_bt_base}/spa2mslg \\\n"
            "    --validate --n_checkpoints {N_CHECKPOINTS}\n"
        )
    )

    cells.append(
        code(
            "# 6.6 - Copy all outputs to Drive + download locally\n"
            "import shutil\n"
            "from pathlib import Path\n"
            "from google.colab import files\n"
            "\n"
            'outputs_dir = PROJECT_ROOT / "outputs"\n'
            "if not outputs_dir.exists() or not any(outputs_dir.iterdir()):\n"
            '    print("No outputs to upload.")\n'
            "else:\n"
            '    for f in sorted(outputs_dir.glob("*.txt")):\n'
            "        dest = DRIVE_SUB / f.name\n"
            "        shutil.copy2(f, dest)\n"
            '        print(f"  Drive: {dest.name}")\n'
            "    print()\n"
            '    print("Downloading all .txt files...")\n'
            '    for f in sorted(outputs_dir.glob("*.txt")):\n'
            "        files.download(str(f))\n"
        )
    )

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
