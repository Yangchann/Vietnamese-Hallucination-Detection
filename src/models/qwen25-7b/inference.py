# inference.py
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.configs.config_qwen25_7b import *

# ========== Load model & tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(INFER_MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
EOS_TOKEN = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    INFER_MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

# ========== Prompt template ==========
def formatting_prompts_func_infer(examples):
    ctxs = examples["context"]
    resps = examples["response"]
    prompts = [BASE_PROMPT.format(ctx=c, ans=a) for c, a in zip(ctxs, resps)]
    return {"prompt": prompts, "completion": [""] * len(prompts)}

# ========== Load dataset ==========
test_dataset = load_dataset("csv", data_files={"test": PRIVATE_TEST_FILE}, split="test")
test_dataset = test_dataset.map(formatting_prompts_func_infer, batched=True)

# ========== Inference ==========
pred_labels = []
ids = test_dataset["id"]

for i in tqdm(range(0, len(test_dataset), INFER_BATCH_SIZE), desc="Predicting"):
    batch = test_dataset[i:i+INFER_BATCH_SIZE]
    prompts = batch["prompt"]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded_batch = tokenizer.batch_decode(
        [out[inputs["input_ids"].shape[1]:] for out in outputs],
        skip_special_tokens=True
    )

    for decoded in decoded_batch:
        pred = "no"
        if "Label:" in decoded:
            pred_text = decoded.split("Label:")[-1].strip().lower()
            for lab in LABEL_LIST:
                if pred_text.startswith(lab):
                    pred = lab
                    break
        else:
            for lab in LABEL_LIST:
                if lab in decoded.lower():
                    pred = lab
                    break
        pred_labels.append(pred)

# ========== Save submit file ==========
df_submit = pd.DataFrame({
    "id": ids,
    "predict_label": pred_labels
})
df_submit.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {OUTPUT_FILE}")
