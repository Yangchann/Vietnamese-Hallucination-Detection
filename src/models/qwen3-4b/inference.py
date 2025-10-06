import os
import torch
import random
import numpy as np
import pandas as pd
import argparse
import importlib.util
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)
import gdown

def load_config_module(config_path: str):
    spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

parser = argparse.ArgumentParser(description="Inference for Qwen models with config")
parser.add_argument("--config", type=str, required=True, help="Path to config.py")
parser.add_argument("--model_name", type=str, required=True, help="Model key or HF id")
parser.add_argument("--test_csv", type=str, default=None, help="Override test csv path")
parser.add_argument("--submit_path", type=str, default=None, help="Override submit csv path")
args = parser.parse_args()

cfg_module = load_config_module(args.config)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(int(getattr(cfg_module, "TRAIN_DEFAULTS", {}).get("seed", 2025)))

resolved = cfg_module.resolve_model_config(args.model_name) if hasattr(cfg_module, "resolve_model_config") else {"model_name": args.model_name, "load_in_8bit": True}

# If model_name is a known checkpoint key, download via gdown to local dir and use that path
ckpt_map = getattr(cfg_module, "CHECKPOINTS", {})
ckpt_dir_root = getattr(cfg_module, "CHECKPOINTS_DIR", "./checkpoints")
os.makedirs(ckpt_dir_root, exist_ok=True)
model_dir_override = None
if args.model_name in ckpt_map:
    url = ckpt_map[args.model_name]
    out_dir = os.path.join(ckpt_dir_root, args.model_name)
    zip_path = os.path.join(ckpt_dir_root, f"{args.model_name}.zip")
    if not os.path.exists(out_dir):
        print(f"Downloading checkpoint for {args.model_name} ...")
        gdown.download(url, zip_path, quiet=False)
        os.system(f"unzip -o \"{zip_path}\" -d \"{out_dir}\"")
    model_dir_override = out_dir

# =============== LOAD DATASET ===============
base_prompt_style = """Instruction:
Given a context and an answer, classify the answer as one of:

- no: Fully consistent with the context, no errors, no extra info.
- intrinsic: Contradicts or misinterprets the context.
- extrinsic: Adds info not in the context and not directly inferable.

Return only: no, intrinsic, or extrinsic.
Context: {ctx}
Answer: {ans}
"""

def format_prompts(examples):
    out = []
    for c, a in zip(examples["context"], examples["response"]):
        out.append(base_prompt_style.format(ctx=c, ans=a))
    return {"prompt": out}

test_csv = args.test_csv or getattr(cfg_module, "DATA_DEFAULTS", {}).get("test_csv")
test_dataset = load_dataset("csv", data_files={"test": test_csv}, split="test")
test_dataset = test_dataset.map(format_prompts, batched=True)
label_list = ["no", "intrinsic", "extrinsic"]

# =============== INFERENCE FUNCTION ===============
def run_inference(model_dir, use_8bit=True):
    quant_cfg = None
    if use_8bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if quant_cfg else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    preds = []
    for sample in tqdm(test_dataset, desc="Predicting"):
        prompt = sample["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()
        pred = next((lab for lab in label_list if lab in decoded), "no")
        preds.append(pred)

    return pd.DataFrame({"id": test_dataset["id"], "predict_label": preds})

use_8bit = bool(resolved.get("load_in_8bit", True))
model_source = model_dir_override if model_dir_override else resolved["model_name"]
df = run_inference(model_source, use_8bit=use_8bit)
submit_path = args.submit_path or getattr(cfg_module, "DATA_DEFAULTS", {}).get("submit_path", "submit.csv")
df.to_csv(submit_path, index=False)
print(f"Saved {submit_path}")

# No ensemble in single-model mode; use external tooling if needed
