import os
import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib.util
from typing import Dict
from datasets import Dataset

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt


def load_config_module(config_path: str):
    spec = importlib.util.spec_from_file_location(
        "dynamic_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


parser = argparse.ArgumentParser(
    description="Inference for LLama models with config")
parser.add_argument("--config", type=str, required=True,
                    help="Path to config.py")
parser.add_argument("--model_name", type=str,
                    required=True, help="Model key or HF id")
parser.add_argument("--test_csv", type=str, default=None,
                    help="Override test csv path")
parser.add_argument("--submit_path", type=str, default=None,
                    help="Override submit csv path")
args = parser.parse_args()

cfg_module = load_config_module(args.config)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(int(getattr(cfg_module, "TRAIN_DEFAULTS", {}).get("seed", 2025)))

ckpt_map = getattr(cfg_module, "CHECKPOINTS", {})
ckpt_dir_root = getattr(cfg_module, "CHECKPOINTS_DIR", "checkpoints")
model_dir = os.path.join(ckpt_dir_root, ckpt_map.get(args.model_name))

# ========================= LOAD MODEL ==========================
model_cfg = getattr(cfg_module, "MODEL_REGISTRY",
                    {}).get(args.model_name, None)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_dir,
    max_seq_length=model_cfg.get("max_seq_length", 2048),
    load_in_4bit=model_cfg.get("load_in_4bit", False),
    load_in_8bit=model_cfg.get("load_in_8bit", True),
    dtype=model_cfg.get("dtype", None),
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer = get_chat_template(
    tokenizer, chat_template=model_cfg.get("chat_template", "llama-3.1"))

# ========================= LOAD AND FORMAT DATASET ========================
base_prompt_style = """
Instruction: Given a context and an answer, classify the answer as one of:
- no: Fully consistent with the context, no errors, no extra info.
- intrinsic: Contradicts or misinterprets the context.
- extrinsic: Adds info not in the context and not directly inferable.

Return only: no, intrinsic, or extrinsic.
Context: {ctx}
Answer: {ans}
"""


def build_conversation_test(row):
    user_prompt = base_prompt_style.format(
        ctx=row["context"],
        ans=row["response"]
    )
    return [{"role": "user", "content": user_prompt}]


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts, }


if args.test_csv is not None:
    print(f"🔍 Using test csv from command line: {args.test_csv}")
    TEST_DATA_PATH = args.test_csv
else:
    print("🔍 Using test csv from config or default")
    TEST_DATA_PATH = getattr(cfg_module, "DATA_DEFAULTS", {}).get(
        "test_csv", "./data/vihallu-private-test.csv")

test_df = pd.read_csv(TEST_DATA_PATH)

test_df["conversations"] = test_df.apply(build_conversation_test, axis=1)
test_dataset = Dataset.from_pandas(test_df[["conversations"]])
test_dataset = standardize_sharegpt(test_dataset).map(
    formatting_prompts_func, batched=True)
# ======================== INFERENCE ==========================

def predict_label(message: Dict, max_new_tokens=5, temperature=0.2, min_p=0.1):
    label_list = ["no", "intrinsic", "extrinsic"]

    inputs = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
    )

    generated_tokens = outputs[0][inputs.shape[1]:]
    decoded = tokenizer.decode(
        generated_tokens, skip_special_tokens=True).strip().lower()

    for label in label_list:
        if label in decoded:
            return label
    print("WARNING: prediction not in labels, defaulting to 'no'")
    return "no"


def run_inference(dataset):
    pred_labels = []
    ids = dataset["id"]

    for sample in tqdm(dataset, desc="Inference"):
        message = sample["conversations"]
        pred = predict_label(message)
        pred_labels.append(pred)

    return ids, pred_labels


ids, pred_labels = run_inference(test_dataset)
submit_path = args.submit_path if args.submit_path is not None else getattr(
    cfg_module, "DATA_DEFAULTS", {}).get("submit_path", "submit.csv")

submit_df = pd.DataFrame({
    "id": ids,
    "predict_label": pred_labels
})

submit_df.to_csv(submit_path, index=False)
print(f"🎯 Inference done! Results saved to {submit_path}")
