import gdown
import torch
import random
import numpy as np
import os
import wandb
import argparse
import importlib.util
from typing import Any, Dict
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import gc
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ================== Download data ==================
# url = "https://drive.google.com/drive/folders/12Hf3nqzJWoDqnE81q2Us3rgy5wR7CdIq"
# output = "./DSC2025"
# gdown.download_folder(url, output=output, quiet=False, use_cookies=False)

def load_config_module(config_path: str):
    spec = importlib.util.spec_from_file_location("dynamic_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Finetune and/or inference Qwen models with config")
parser.add_argument("--config", type=str, required=True, help="Path to config.py")
parser.add_argument("--model_name", type=str, required=True, help="Model key or HF id")
parser.add_argument("--mode", type=str, default="train", choices=["train", "inference", "train_and_infer"], help="Run mode")
parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
parser.add_argument("--train_csv", type=str, default=None, help="Override train csv path")
parser.add_argument("--test_csv", type=str, default=None, help="Override test csv path")
parser.add_argument("--submit_path", type=str, default=None, help="Override submit csv path")
args = parser.parse_args()

cfg_module = load_config_module(args.config)

seed = int(getattr(cfg_module, "TRAIN_DEFAULTS", {}).get("seed", 2025))
set_seed(seed)

# ================== Login ==================
if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(hf_token)

resolved = cfg_module.resolve_model_config(args.model_name) if hasattr(cfg_module, "resolve_model_config") else {"model_name": args.model_name, "load_in_8bit": True, "torch_dtype": "float16"}

bnb_config = None
if resolved.get("load_in_8bit", True):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

model_dir = resolved["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16 if bnb_config is not None else None,
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
EOS_TOKEN = tokenizer.eos_token

# ================== Prompt templates ==================
base_prompt_style = """Instruction:  
Given a context and an answer, classify the answer as one of:  

- no: Fully consistent with the context, no errors, no extra info.  
- intrinsic: Contradicts or misinterprets the context.  
- extrinsic: Adds info not in the context and not directly inferable.  

Return only: no, intrinsic, or extrinsic.
Context: {ctx}
Answer: {ans}
"""

def formatting_prompts_func_train(examples):
    contexts = examples["context"]
    responses = examples["response"]
    labels = examples["label"]
    prompts_out, completions_out = [], []
    for ctx, ans, lab in zip(contexts, responses, labels):
        prompt_text = base_prompt_style.format(ctx=ctx, ans=ans)
        completion_text = "Label: " + lab + EOS_TOKEN
        prompts_out.append(prompt_text)
        completions_out.append(completion_text)
    return {"prompt": prompts_out, "completion": completions_out}

def formatting_prompts_func_infer(examples):
    contexts = examples["context"]
    responses = examples["response"]
    prompts_out, completions_out = [], []
    for ctx, ans in zip(contexts, responses):
        prompt_text = base_prompt_style.format(ctx=ctx, ans=ans)
        prompts_out.append(prompt_text)
        completions_out.append("")  # không có nhãn
    return {"prompt": prompts_out, "completion": completions_out}

# ================== Load dataset ==================
train_csv = args.train_csv or getattr(cfg_module, "DATA_DEFAULTS", {}).get("train_csv", "./DSC2025/final_data.csv")
test_csv = args.test_csv or getattr(cfg_module, "DATA_DEFAULTS", {}).get("test_csv", "./DSC2025/vihallu-private-test.csv")

train_dataset = load_dataset("csv", data_files={"train": train_csv}, split="train")
test_dataset = load_dataset("csv", data_files={"test": test_csv}, split="test")

# Split train/dev
train_test_split = train_dataset.train_test_split(test_size=0.1, seed=seed)
train_data = train_test_split["train"]
dev_data = train_test_split["test"]

# Apply formatting
train_data = train_data.map(formatting_prompts_func_train, batched=True)
dev_data = dev_data.map(formatting_prompts_func_train, batched=True)
test_dataset = test_dataset.map(formatting_prompts_func_infer, batched=True)

print(f'{train_data.shape=}')
print(f'{dev_data.shape=}')
print(f'{test_dataset.shape=}')

# ================== Data collator ==================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# ================== LoRA config ==================
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, peft_config)

# ================== Eval function (custom) ==================
label_list = ["no", "intrinsic", "extrinsic"]

def eval_like_inference(model, tokenizer, dataset, label_list, max_new_tokens=10):
    preds, trues = [], []
    for sample in dataset:
        prompt = sample["prompt"]
        true_label = sample["completion"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # --- pred ---
        pred = "no"
        if "Label:" in decoded:
            pred_text = decoded.split("Label:")[-1].strip().lower()
            for lab in label_list:
                if pred_text.startswith(lab):
                    pred = lab
                    break
        else:
            for lab in label_list:
                if lab in decoded.lower():
                    pred = lab
                    break

        # --- true ---
        true = "no"
        if "Label:" in true_label:
            text = true_label.split("Label:")[-1].strip().lower()
            for lab in label_list:
                if text.startswith(lab):
                    true = lab
                    break

        preds.append(pred)
        trues.append(true)

    f1 = f1_score(trues, preds, average="micro")
    acc = accuracy_score(trues, preds)
    print("Eval Accuracy:", acc)
    print("Eval Micro F1:", f1)
    return {"accuracy": acc, "f1": f1}

# ================== Custom Trainer ==================
class CustomSFTTrainer(SFTTrainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        metrics = eval_like_inference(self.model, tokenizer, eval_dataset, label_list)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        print(f"Step {self.state.global_step}: {metrics}")
        print(f"Best metric so far: {getattr(self.state, 'best_metric', None)}")
        return metrics

td = getattr(cfg_module, "TRAIN_DEFAULTS", {})
output_dir = args.output_dir or getattr(cfg_module, "DATA_DEFAULTS", {}).get("output_dir", "output_trainfull")
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=int(td.get("per_device_train_batch_size", 1)),
    per_device_eval_batch_size=int(td.get("per_device_eval_batch_size", 1)),
    gradient_accumulation_steps=int(td.get("gradient_accumulation_steps", 2)),
    optim="paged_adamw_32bit",
    num_train_epochs=int(td.get("num_train_epochs", 5)),
    logging_steps=int(td.get("logging_steps", 100)),
    warmup_steps=int(td.get("warmup_steps", 100)),
    lr_scheduler_type="linear",
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=int(td.get("eval_steps", 2100)),
    learning_rate=float(td.get("learning_rate", 2e-5)),
    fp16=True,
    group_by_length=True,
    save_strategy="steps",
    save_steps=int(td.get("save_steps", 2100)),
    report_to="wandb",
    run_name="qwen3-vihallu",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True
)

# ================== Trainer ==================
trainer = CustomSFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=dev_data,
    peft_config=peft_config,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

def maybe_train():
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = False
    trainer.train()

should_train = args.mode in ("train", "train_and_infer")
if should_train:
    maybe_train()


def run_inference(current_model, current_tokenizer, dataset):
    label_list = ["no", "intrinsic", "extrinsic"]
    pred_labels = []
    ids = dataset["id"]

    for sample in tqdm(dataset, desc="Predicting"):
        prompt = sample["prompt"]
        inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
        outputs = current_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=current_tokenizer.eos_token_id,
        )
        decoded = current_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        pred = "no"
        if "Label:" in decoded:
            pred_text = decoded.split("Label:")[-1].strip().lower()
            for lab in label_list:
                if pred_text.startswith(lab):
                    pred = lab
                    break
        else:
            dl = decoded.lower()
            for lab in label_list:
                if lab in dl:
                    pred = lab
                    break
        pred_labels.append(pred)

    return ids, pred_labels

should_infer = args.mode in ("inference", "train_and_infer", "train")
if should_infer:
    best_model = trainer.model if should_train else model
    best_tokenizer = tokenizer
    ids, pred_labels = run_inference(best_model, best_tokenizer, test_dataset)
    submit_path = args.submit_path or getattr(cfg_module, "DATA_DEFAULTS", {}).get("submit_path", "submit_train_full.csv")
    df_submit = pd.DataFrame({"id": ids, "predict_label": pred_labels})
    df_submit.to_csv(submit_path, index=False)
    print(f"Saved {submit_path}")
