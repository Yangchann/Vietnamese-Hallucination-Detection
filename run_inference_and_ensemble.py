import os
import pandas as pd
import subprocess
import random

# ========== 1. Run inference for Llama models ==========
llama_cmds = [
    "python src/models/Llama/inference.py --config src/configs/config_llama.py --model_name Llama32-3B-4bit --test_csv data/vihallu-private-test.csv --submit_path submit-Llama32-3B-4bit.csv",
    "python src/models/Llama/inference.py --config src/configs/config_llama.py --model_name Llama32-3B-16bit --test_csv data/vihallu-private-test.csv --submit_path submit-Llama32-3B-16bit.csv",
    "python src/models/Llama/inference.py --config src/configs/config_llama.py --model_name Llama2-7B-16bit --test_csv data/vihallu-private-test.csv --submit_path submit-Llama2-7B-16bit.csv",
]

print("🔹 Running Llama inferences...")
for cmd in llama_cmds:
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# ========== 2. Ensemble Llama models ==========
print("🔹 Ensemble Llama models...")
f1 = pd.read_csv("submit-Llama32-3B-4bit.csv")
f2 = pd.read_csv("submit-Llama32-3B-16bit.csv")
f3 = pd.read_csv("submit-Llama2-7B-16bit.csv")


def major_vote_llama(row):
    preds = [row["predict_label_x"],
             row["predict_label_y"], row["predict_label"]]
    if len(set(preds)) == 3:
        return row["predict_label_y"]
    else:
        return max(preds, key=preds.count)


llama_ensemble = f1.merge(
    f2, on="id", suffixes=("_x", "_y")).merge(f3, on="id")
llama_ensemble["predict_label"] = llama_ensemble.apply(
    major_vote_llama, axis=1)
llama_ensemble[["id", "predict_label"]].to_csv(
    "submit-Llama-Ensemble.csv", index=False)
print("✅ Saved submit-Llama-Ensemble.csv")

# ========== 3. Run inference for Qwen2.5-7B ==========
print("🔹 Running Qwen2.5-7B inference...")
qwen25_cmd = "python src/models/qwen25-7b/inference.py --config src/configs/config_qwen25_7b.py --model_name qwen25-7b --test_csv data/vihallu-private-test.csv --submit_path submit-qwen25-7b.csv"
subprocess.run(qwen25_cmd, shell=True, check=True)
print("✅ Saved submit-qwen25-7b.csv")

# ========== 4. Run inference for Qwen3 models ==========
qwen3_cmds = [
    "python src/models/qwen3-4b/inference.py --config src/configs/config_qwen3_4b.py --model_name qwen3-4b-instruct-2507-int8 --test_csv data/vihallu-private-test.csv --submit_path submit-qwen3-4b-instruct-2507-int8.csv",
    "python src/models/qwen3-4b/inference.py --config src/configs/config_qwen3_4b.py --model_name qwen3-4b-instruct-2507-full --test_csv data/vihallu-private-test.csv --submit_path submit-qwen3-4b-instruct-2507-full.csv",
    "python src/models/qwen3-4b/inference.py --config src/configs/config_qwen3_4b.py --model_name qwen3-4b-thinking-2507-int8 --test_csv data/vihallu-private-test.csv --submit_path submit-qwen3-4b-thinking-2507-int8.csv",
]

print("🔹 Running Qwen3 inferences...")
for cmd in qwen3_cmds:
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# ========== 5. Ensemble Qwen3 models ==========
print("🔹 Ensemble Qwen3 models...")
q1 = pd.read_csv("submit-qwen3-4b-instruct-2507-int8.csv")
q2 = pd.read_csv("submit-qwen3-4b-instruct-2507-full.csv")
q3 = pd.read_csv("submit-qwen3-4b-thinking-2507-int8.csv")


def major_vote_qwen(row):
    preds = [row["predict_label_x"],
             row["predict_label_y"], row["predict_label"]]
    if len(set(preds)) == 3:
        return random.choice(preds)
    else:
        return max(preds, key=preds.count)


qwen3_ensemble = q1.merge(
    q2, on="id", suffixes=("_x", "_y")).merge(q3, on="id")
qwen3_ensemble["predict_label"] = qwen3_ensemble.apply(major_vote_qwen, axis=1)
qwen3_ensemble[["id", "predict_label"]].to_csv(
    "submit-Qwen3-Ensemble.csv", index=False)
print("✅ Saved submit-Qwen3-Ensemble.csv")

# ========== 6. Final Ensemble (Llama + Qwen3 + Qwen2.5) ==========
print("🔹 Final Ensemble (Llama + Qwen3 + Qwen2.5)...")
qwen25 = pd.read_csv("submit-qwen25-7b.csv")

final = llama_ensemble.merge(qwen3_ensemble, on="id", suffixes=(
    "_llama", "_qwen3")).merge(qwen25, on="id")


def final_vote(row):
    preds = [row["predict_label_llama"],
             row["predict_label_qwen3"], row["predict_label"]]
    if len(set(preds)) == 3:
        return random.choice(preds)
    else:
        return max(preds, key=preds.count)


final["predict_label"] = final.apply(final_vote, axis=1)
final[["id", "predict_label"]].to_csv("submit.csv", index=False)
print("✅ Saved submit.csv")

print("\n🎯 DONE — All inference & ensemble completed!")
