import os

# Centralized configuration for models, training, and inference


def get_default_seed() -> int:
    return int(os.environ.get("SEED", 2025))


# Model registry: define how to load each model
MODEL_REGISTRY = {
    # Qwen/Qwen3-4B-Instruct-2507 in 8-bit
    "qwen3-4b-instruct-2507-int8": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "load_in_8bit": True,
        "torch_dtype": "float16",
    },
    # Qwen/Qwen3-4B-Instruct-2507 full precision
    "qwen3-4b-instruct-2507-full": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "load_in_8bit": False,
        "torch_dtype": "float16",
    },
    # Qwen/Qwen3-4B-Thinking-2507 in 8-bit
    "qwen3-4b-thinking-2507-int8": {
        "model_name": "Qwen/Qwen3-4B-Thinking-2507",
        "load_in_8bit": True,
        "torch_dtype": "float16",
    },
}


# Default training hyperparameters
TRAIN_DEFAULTS = {
    "num_train_epochs": 5,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 100,
    "eval_steps": 2100,
    "save_steps": 2100,
    "logging_steps": 100,
    "seed": get_default_seed(),
}


# Data paths
DATA_DEFAULTS = {
    "train_csv": "./data/vihallu-train.csv",
    "test_csv": "./data/vihallu-private-test.csv",
    "output_dir": "output_trainfull",
    "submit_path": "submit_train_full.csv",
}


# Optional pretrained checkpoints for inference-only (downloadable via gdown)
CHECKPOINTS = {
    "qwen3-4b-instruct-2507-int8": "Qwen3-4B-Instruction-8bit",
    "qwen3-4b-thinking-2507-int8": "Qwen3-4B-Instruction-full",
    "qwen3-4b-instruct-2507-full": "Qwen3-4B-Thinking",
}

# Where to store downloaded checkpoints
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints")


def resolve_model_config(model_key: str):
    if model_key in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[model_key].copy()
        cfg["key"] = model_key
        return cfg
    # allow passing raw HF model id
    return {
        "key": model_key,
        "model_name": model_key,
        "load_in_8bit": True,
        "torch_dtype": "float16",
    }


