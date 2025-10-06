#!/usr/bin/env bash
# ============================================================
#   Vietnamese Hallucination Detection — Full Run Script
#   Usage: bash RUN_REFERENCE_AND_ENSEMBLE.sh [CONFIG_PATH]
# ============================================================

set -euo pipefail

# -----------------------------
# CONFIG PATHS
# -----------------------------
LLAMA_CONFIG_PATH="./src/configs/config_llama.py"
QWEN3_CONFIG_PATH="./src/configs/config_qwen3-4b.py"
QWEN25_CONFIG_PATH="./src/configs/config_qwen25-7b.py"

ROOT_DIR=$(dirname "$0")
cd "$ROOT_DIR"

RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# ============================================================
# 1) LLaMA Inference
# ============================================================
echo "🦙 [1/6] Running inference for LLaMA models..."
pushd src/models/llama > /dev/null

python ./inference.py --config "${LLAMA_CONFIG_PATH}" --model_name Llama3.2-3B-4bit  --submit_path "${RESULTS_DIR}/llama_Llama3.2-3B-4bit.csv"
python ./inference.py --config "${LLAMA_CONFIG_PATH}" --model_name Llama3.2-3B-16bit --submit_path "${RESULTS_DIR}/llama_Llama3.2-3B-16bit.csv"
python ./inference.py --config "${LLAMA_CONFIG_PATH}" --model_name Llama2-7B-16bit  --submit_path "${RESULTS_DIR}/llama_Llama2-7B-16bit.csv"

popd > /dev/null


# ============================================================
# 2) Qwen3 Inference
# ============================================================
echo "🧬 [2/6] Running inference for Qwen3 models..."
pushd src/models/qwen3-4b > /dev/null

python ./inference.py --config "${QWEN3_CONFIG_PATH}" --model_name qwen3-4b-instruct-2507-int8 --submit_path "${RESULTS_DIR}/qwen3-4b-1.csv"
python ./inference.py --config "${QWEN3_CONFIG_PATH}" --model_name qwen3-4b-instruct-2507-full  --submit_path "${RESULTS_DIR}/qwen3-4b-2.csv"
python ./inference.py --config "${QWEN3_CONFIG_PATH}" --model_name qwen3-4b-thinking-2507-int8  --submit_path "${RESULTS_DIR}/qwen3-4b-3.csv"

popd > /dev/null


# ============================================================
# 3) Qwen2.5 Inference
# ============================================================
echo "⚡ [3/6] Running inference for Qwen2.5 model..."
pushd src/models/qwen25-7b > /dev/null
python ./inference.py
popd > /dev/null


# ============================================================
# 4) First-level Ensemble (within each model family)
# ============================================================
echo "🧩 [4/6] Running first-level ensemble..."
python - <<'PY'
import pandas as pd
from collections import Counter
from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

def ensemble_majority_vote(csv_paths, out_path, fallback=None):
    dfs = [pd.read_csv(p) for p in csv_paths]
    merged = dfs[0][["id"]].copy()
    for i, df in enumerate(dfs):
        merged[f"pred_{i+1}"] = df["predict_label"]

    preds = []
    for _, row in merged.iterrows():
        votes = [row[f"pred_{i+1}"] for i in range(len(dfs))]
        most_common = Counter(votes).most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            preds.append(fallback or votes[0])  # fallback if tie
        else:
            preds.append(most_common[0][0])

    out_df = pd.DataFrame({"id": merged["id"], "predict_label": preds})
    out_df.to_csv(out_path, index=False)
    print(f"✅ Ensemble saved to {out_path}")

# LLaMA ensemble
ensemble_majority_vote([
    "results/llama_Llama3.2-3B-4bit.csv",
    "results/llama_Llama3.2-3B-16bit.csv",
    "results/llama_Llama2-7B-16bit.csv",
], "results/llama_ensemble.csv")

# Qwen3 ensemble
ensemble_majority_vote([
    "results/qwen3-4b-1.csv",
    "results/qwen3-4b-2.csv",
    "results/qwen3-4b-3.csv",
], "results/qwen3_ensemble.csv")
PY


# ============================================================
# 5) Second-level Ensemble (cross-family)
# ============================================================
echo "🏁 [5/6] Running second-level ensemble..."
python - <<'PY'
import pandas as pd
from collections import Counter

def ensemble_final(files, output_path, fallback=None):
    dfs = [pd.read_csv(f) for f in files]
    merged = dfs[0][["id"]].copy()
    for i, df in enumerate(dfs):
        merged[f"pred_{i+1}"] = df["predict_label"]

    preds = []
    for _, row in merged.iterrows():
        votes = [row[f"pred_{i+1}"] for i in range(len(dfs))]
        most_common = Counter(votes).most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            preds.append(fallback or votes[0])
        else:
            preds.append(most_common[0][0])

    out_df = pd.DataFrame({"id": merged["id"], "predict_label": preds})
    out_df.to_csv(output_path, index=False)
    print(f"✅ Final ensemble saved to {output_path}")

ensemble_final([
    "results/llama_ensemble.csv",
    "results/qwen3_ensemble.csv",
    "results/qwen25-7b.csv",
], "results/final_ensemble.csv")
PY


# ============================================================
# 6) Done
# ============================================================
echo "🎉 [6/6] All tasks completed successfully!"
echo "Final result available at: results/final_ensemble.csv"
