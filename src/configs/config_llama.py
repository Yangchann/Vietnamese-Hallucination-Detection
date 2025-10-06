import os
import torch

def get_default_seed() -> int:
    return int(os.environ.get("SEED", 2025))

MODEL_REGISTRY = {
    # Llama-3.2-3B in 4-bit
    "Llama32-3B-4bit": {
        "model_name": "unsloth/Llama-3.2-3B-Instruct",
        "load_in_4bit": True,
        "load_in_8bit": False,
        "dtype": torch.float16,
        "chat_template": "llama-3.1",
        "max_seq_length": 2560,
    },
    # Llama-3.2-3B in 16-bit
    "Llama32-3B-16bit": {
        "model_name": "unsloth/Llama-3.2-3B-Instruct",
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": torch.float16,
        "chat_template": "llama-3.1",
        "max_seq_length": 2560,
    },
    # Llama-2-7B in 16-bit
    "Llama2-7B-16bit": {
        "model_name": "meta-llama/Llama-2-7b",
        "load_in_4bit": False,
        "load_in_8bit": False,
        "dtype": torch.float16,
        "chat_template": "llama",
        "max_seq_length": 2048,
    },
}


TRAIN_DEFAULTS = {
    "num_train_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "eval_strategy": "steps",
    "eval_steps": 50,
    "logging_strategy": "steps",
    "logging_steps": 10,
    "optim": "adamw_torch_fused",
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "report_to": "wandb",
    "seed": get_default_seed(),
    "stopping_patience": 3,
}

DATA_DEFAULTS = {
    "train_csv": "./data/vihallu-train.csv",
    "test_csv": "./data/vihallu-private-test.csv",
    "output_dir": "output_train",
    "submit_path": "submit.csv",
}

CHECKPOINTS = {
    "Llama32-3B-4bit": "Llama3.2-3B-Instruct-4bit",
    "Llama32-3B-16bit": "Llama32-3B-Instruct-16bit",
    "Llama2-7B-16bit": "Llama2-7B-16bit",
}

CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
