# config.py
import torch
import random
import numpy as np
import os

# ========== Seed ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 2025
set_seed(SEED)

# ========== Model & Tokenizer ==========
MODEL_DIR = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = "place hf token"
WANDB_KEY = "place wandb key"

# ========== Dataset ==========
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/vihallu-train.csv"
TEST_FILE = f"{DATA_DIR}/vihallu-public-test.csv"
TEST_SIZE = 0.1

# ========== Training Hyperparams ==========
EVAL_BATCH_SIZE = 8
MAX_NEW_TOKENS = 10

TRAINING_ARGS = dict(
    output_dir="output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=10,
    logging_steps=100,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=2100,
    learning_rate=1e-4,
    fp16=True,
    group_by_length=True,
    save_strategy="steps",
    save_steps=2100,
    report_to="wandb",
    run_name="qwen3-vihallu",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
)

# ========== Prompt Template ==========
BASE_PROMPT = """Instruction:  
You are given a CONTEXT and an ANSWER. Classify their relationship into exactly one of:

- no: ANSWER is fully consistent with CONTEXT, with no errors or extra information.
- intrinsic: ANSWER contradicts or misinterprets the CONTEXT.
- extrinsic: ANSWER introduces information not present in the CONTEXT and not directly inferable.

Return only: no, intrinsic, or extrinsic.
Context: {ctx}
Answer: {ans}
"""

LABEL_LIST = ["no", "intrinsic", "extrinsic"]
EOS_TOKEN = None  # sẽ gán sau khi load tokenizer

# ========== Inference ==========
INFER_MODEL_DIR = "./checkpoints/Qwen2.5-7B"  # model đã fine-tuned hoặc base
PRIVATE_TEST_FILE = f"{DATA_DIR}/vihallu-private-test.csv"
INFER_BATCH_SIZE = 4
MAX_NEW_TOKENS = 10
OUTPUT_FILE = "submit-Qwen2.5-7B.csv"