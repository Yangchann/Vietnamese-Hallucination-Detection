import gc
import os
import torch
import random
import argparse
import importlib.util
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict
from datasets import Dataset


import wandb
from huggingface_hub import login
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback

RANDOM_STATE = 42

def load_config_module(config_path: str):
    spec = importlib.util.spec_from_file_location(
        "dynamic_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(
    description="Finetune and/or inference LLama models with config")
parser.add_argument("--config", type=str, required=True,
                    help="Path to config.py")
parser.add_argument("--model_name", type=str,
                    required=True, help="Model key or HF id")
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "inference", "train_and_infer"], help="Run mode")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Override output dir")
parser.add_argument("--train_csv", type=str, default=None,
                    help="Override train csv path")
parser.add_argument("--test_csv", type=str, default=None,
                    help="Override test csv path")
parser.add_argument("--submit_path", type=str, default=None,
                    help="Override submit csv path")
args = parser.parse_args()

cfg_module = load_config_module(args.config)

MODEL_REGISTRY = cfg_module.MODEL_REGISTRY
TRAIN_DEFAULTS = cfg_module.TRAIN_DEFAULTS
DATA_DEFAULTS = cfg_module.DATA_DEFAULTS
CHECKPOINTS = getattr(cfg_module, "CHECKPOINTS", {})
CHECKPOINTS_DIR = getattr(cfg_module, "CHECKPOINTS_DIR", "checkpoints")

seed = int(getattr(cfg_module, "TRAIN_DEFAULTS", {}).get("seed", 2025))
set_seed(seed)

# ===============================
#   Setting Model and Tokenizer
# ===============================
if args.model_name not in MODEL_REGISTRY:
    raise ValueError(f"Model {args.model_name} not found in MODEL_REGISTRY")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_REGISTRY[args.model_name]["model_name"],
    max_seq_length=MODEL_REGISTRY[args.model_name]["max_seq_length"],
    dtype=MODEL_REGISTRY[args.model_name]["dtype"],
    load_in_4bit=MODEL_REGISTRY[args.model_name]["load_in_4bit"],
    load_in_8bit=MODEL_REGISTRY[args.model_name]["load_in_8bit"],
    device_map="auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_STATE,
    use_rslora=False,
    loftq_config=None,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# ===============================
#     Prompt Template
# ===============================

base_prompt_style = """
Instruction: Given a context and an answer, classify the answer as one of:
- no: Fully consistent with the context, no errors, no extra info.
- intrinsic: Contradicts or misinterprets the context.
- extrinsic: Adds info not in the context and not directly inferable.

Return only: no, intrinsic, or extrinsic.
Context: {ctx}
Answer: {ans}
"""


def build_conversation_train(row):
    user_prompt = base_prompt_style.format(
        prompt=row["prompt"],
        ctx=row["context"],
        ans=row["response"]
    )
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": row["label"]},
    ]


def build_conversation_test(row):
    user_prompt = base_prompt_style.format(
        prompt=row["prompt"],
        ctx=row["context"],
        ans=row["response"]
    )
    return [{"role": "user", "content": user_prompt}]


# ===============================
#     Load and Format Dataset
# ===============================

train_csv = args.train_csv if args.train_csv is not None else DATA_DEFAULTS.get(
    "train_csv", "./data/vihallu-train.csv")
test_csv = args.test_csv if args.test_csv is not None else DATA_DEFAULTS.get(
    "test_csv", "./data/vihallu-private-test.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
train_df, dev_df = train_test_split(
    train_df, test_size=0.1, random_state=RANDOM_STATE, shuffle=True, stratify=train_df["label"])

train_df["conversations"] = train_df.apply(build_conversation_train, axis=1)
dev_df["conversations"] = dev_df.apply(build_conversation_train, axis=1)
test_df["conversations"] = test_df.apply(build_conversation_test, axis=1)

train_dataset = Dataset.from_pandas(train_df[["id", "conversations"]])
dev_dataset = Dataset.from_pandas(dev_df[["id", "conversations"]])
test_dataset = Dataset.from_pandas(test_df[["id", "conversations"]])

# Formatting data


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts, }


train_dataset = standardize_sharegpt(train_dataset).map(
    formatting_prompts_func, batched=True,)
dev_dataset = standardize_sharegpt(dev_dataset).map(
    formatting_prompts_func, batched=True,)
test_dataset = standardize_sharegpt(test_dataset).map(
    formatting_prompts_func, batched=True,)

# ===============================
#        Training
# ===============================
mode = args.mode
OUTPUT_DIR = args.output_dir if args.output_dir is not None else DATA_DEFAULTS.get(
    "output_dir", "output_train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    dataset_text_field="text",
    max_seq_length=MODEL_REGISTRY[args.model_name]["max_seq_length"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=True,
    args=SFTConfig(
        fp16=True,
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_DEFAULTS.get("num_train_epochs", 3),
        per_device_train_batch_size=TRAIN_DEFAULTS.get(
            "per_device_train_batch_size", 1),
        per_device_eval_batch_size=TRAIN_DEFAULTS.get(
            "per_device_eval_batch_size", 1),
        gradient_accumulation_steps=TRAIN_DEFAULTS.get(
            "gradient_accumulation_steps", 1),
        eval_strategy=TRAIN_DEFAULTS.get("eval_strategy", "steps"),
        eval_steps=TRAIN_DEFAULTS.get("eval_steps", 100),
        learning_rate=TRAIN_DEFAULTS.get("learning_rate", 5e-5),
        weight_decay=TRAIN_DEFAULTS.get("weight_decay", 0.0),
        warmup_ratio=TRAIN_DEFAULTS.get("warmup_ratio", 0.0),
        lr_scheduler_type=TRAIN_DEFAULTS.get("lr_scheduler_type", "linear"),
        logging_strategy=TRAIN_DEFAULTS.get("logging_strategy", "steps"),
        logging_steps=TRAIN_DEFAULTS.get("logging_steps", 10),
        optim=TRAIN_DEFAULTS.get("optim", "adamw_torch_fused"),
        seed=TRAIN_DEFAULTS.get("seed", 42),
        save_strategy=TRAIN_DEFAULTS.get("save_strategy", "steps"),
        save_steps=TRAIN_DEFAULTS.get("save_steps", 100),
        save_total_limit=TRAIN_DEFAULTS.get("save_total_limit", 1),
        load_best_model_at_end=TRAIN_DEFAULTS.get(
            "load_best_model_at_end", True),
        report_to=TRAIN_DEFAULTS.get("report_to", "wandb"),
    ),
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=TRAIN_DEFAULTS.get("stopping_patience", 3))],

)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)


def maybe_train():
    if mode in ["train", "train_and_infer"]:
        print("🚀 Start training ...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        gc.collect()
        torch.cuda.empty_cache()
        model.config.use_cache = False
        trainer.train()


should_train = args.mode in ["train", "train_and_infer"]
if should_train:
    maybe_train()


def run_inference(model, tokenizer, dataset):

    label_list = ["no", "intrinsic", "extrinsic"]
    pred_labels = []
    ids = dataset["id"]

    for sample in tqdm(dataset, desc="Inference"):
        message = sample["conversations"]
        inputs = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=5,
            temperature=0.2,
            min_p=0.1,
        )

        generated_tokens = outputs[0][inputs.shape[1]:]
        decoded = tokenizer.decode(
            generated_tokens, skip_special_tokens=True).strip().lower()

        pred = "no"
        for label in label_list:
            if label in decoded:
                pred = label
                break
        pred_labels.append(pred)

    return ids, pred_labels


should_infer = args.mode in ["inference", "train_and_infer"]
if should_infer:
    best_model = trainer.model if should_train else model
    best_tokenizer = tokenizer
    ids, pred_labels = run_inference(best_model, best_tokenizer, test_dataset)
    submit_path = args.submit_path if args.submit_path is not None else DATA_DEFAULTS.get("submit_path", "submit.csv")
    submit_df = pd.DataFrame({
        "id": ids,
        "predict_label": pred_labels,
    })
    submit_df.to_csv(submit_path, index=False)
    print(f"🎯 Submission saved to {submit_path}")
