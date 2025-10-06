## Training and Inference with Configurable Qwen Models

This repo now supports configurable finetuning and inference using a centralized `code/config.py`.

### Supported models
- qwen3-4b-instruct-2507-int8 (Qwen/Qwen3-4B-Instruct-2507, 8-bit)
- qwen3-4b-instruct-2507-full (Qwen/Qwen3-4B-Instruct-2507, full precision)
- qwen3-4b-thinking-2507-int8 (Qwen/Qwen3-4B-Thinking-2507, 8-bit)

You can also pass any raw HF model id via `--model_name`.

### Files
- `code/config.py`: model registry, train/infer defaults, data paths.
- `code/finetune.py`: finetune and optional inference in one run.
- `code/inference.py`: inference-only entrypoint.

### Environment
Optionally set tokens via environment variables:
```bash
set HF_TOKEN=hf_...               # Windows PowerShell: $env:HF_TOKEN="hf_..."
set WANDB_API_KEY=...             # Windows PowerShell: $env:WANDB_API_KEY="..."
```

### Finetune
```bash
python code/finetune.py --config code/config.py \
  --model_name qwen3-4b-instruct-2507-int8 \
  --mode train \
  --output_dir output_trainfull
```

Train and then run inference:
```bash
python code/finetune.py --config code/config.py \
  --model_name qwen3-4b-instruct-2507-full \
  --mode train_and_infer \
  --train_csv ./DSC2025/final_data.csv \
  --test_csv ./DSC2025/vihallu-private-test.csv \
  --submit_path submit_train_full.csv
```

### Inference
```bash
python code/inference.py --config code/config.py \
  --model_name qwen3-4b-thinking-2507-int8 \
  --test_csv ./DSC2025/vihallu-private-test.csv \
  --submit_path submit.csv
```

### Notes
- Modify `code/config.py` to adjust model registry or defaults.
- When passing a raw HF model id to `--model_name`, it will default to 8-bit loading unless changed in code.

