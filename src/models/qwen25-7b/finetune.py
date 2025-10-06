# finetune.py
import sys
import gc
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import login

from src.configs.config_qwen25_7b import * 

# ========== Logging ==========
sys.stdout = open("train_log.txt", "w", buffering=1)
sys.stderr = sys.stdout

print("===== Start Fine-tuning =====")

# ========== Auth ==========
wandb.login(key=WANDB_KEY, relogin=True)
login(HF_TOKEN)

# ========== Model & Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
EOS_TOKEN = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# ========== Prompt formatting ==========
def formatting_prompts_func_train(examples):
    ctxs, ans, labs = examples["context"], examples["response"], examples["label"]
    prompts, completions = [], []
    for c, a, l in zip(ctxs, ans, labs):
        p = BASE_PROMPT.format(ctx=c, ans=a)
        completions.append("Label: " + l + EOS_TOKEN)
        prompts.append(p)
    return {"prompt": prompts, "completion": completions}

def formatting_prompts_func_infer(examples):
    ctxs, ans = examples["context"], examples["response"]
    prompts = [BASE_PROMPT.format(ctx=c, ans=a) for c, a in zip(ctxs, ans)]
    return {"prompt": prompts, "completion": [""] * len(prompts)}

# ========== Load Dataset ==========
train_dataset = load_dataset("csv", data_files={"train": TRAIN_FILE}, split="train")
test_dataset = load_dataset("csv", data_files={"test": TEST_FILE}, split="test")

split_data = train_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
train_data, dev_data = split_data["train"], split_data["test"]

train_data = train_data.map(formatting_prompts_func_train, batched=True)
dev_data = dev_data.map(formatting_prompts_func_train, batched=True)
test_dataset = test_dataset.map(formatting_prompts_func_infer, batched=True)

# ========== Data Collator ==========
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# ========== LoRA ==========
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)

# ========== Eval Function ==========
def eval_like_inference_batched(model, tokenizer, dataset, label_list):
    model.eval()
    preds, trues = [], []
    for i in range(0, len(dataset), EVAL_BATCH_SIZE):
        batch = dataset[i:i+EVAL_BATCH_SIZE]
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        decoded = tokenizer.batch_decode([o[inputs["input_ids"].shape[1]:] for o in outputs], skip_special_tokens=True)
        for d, t in zip(decoded, batch["completion"]):
            pred = next((l for l in label_list if l in d.lower()), "no")
            true = next((l for l in label_list if l in t.lower()), "no")
            preds.append(pred)
            trues.append(true)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="micro")
    return {"accuracy": acc, "f1": f1}

# ========== Custom Trainer ==========
class CustomSFTTrainer(SFTTrainer):
    def evaluate(self, eval_dataset=None, **kwargs):
        dataset = eval_dataset or self.eval_dataset
        metrics = eval_like_inference_batched(self.model, tokenizer, dataset, LABEL_LIST)
        self.log({f"eval_{k}": v for k, v in metrics.items()})
        return metrics

# ========== Training ==========
args = TrainingArguments(**TRAINING_ARGS)
trainer = CustomSFTTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=dev_data,
    peft_config=peft_config,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

gc.collect()
torch.cuda.empty_cache()
trainer.train()
