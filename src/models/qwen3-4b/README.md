## Training and Inference with Configurable Qwen Models

This repo now supports configurable finetuning and inference using a centralized `code/config.py`.

### Supported models

- qwen3-4b-instruct-2507-int8 (Qwen/Qwen3-4B-Instruct-2507, 8-bit)
- qwen3-4b-instruct-2507-full (Qwen/Qwen3-4B-Instruct-2507, full precision)
- qwen3-4b-thinking-2507-int8 (Qwen/Qwen3-4B-Thinking-2507, 8-bit)

You can also pass any raw HF model id via `--model_name`.

### Environment

Optionally set tokens via environment variables:

```bash
set HF_TOKEN=hf_...               # Windows PowerShell: $env:HF_TOKEN="hf_..."
set WANDB_API_KEY=...             # Windows PowerShell: $env:WANDB_API_KEY="..."
```

### Finetune

```bash
python src/models/qwen3-4b/finetune.py --config src/configs/config_qwen3_4b.py \
  --model_name qwen3-4b-instruct-2507-int8 \
  --mode train \
  --output_dir output
```

### Inference

```bash
python src/models/qwen3-4b/inference.py --config src/configs/config_qwen3_4b.py \
  --model_name qwen3-4b-instruct-2507-int8 \
  --test_csv data/exemple_test.csv \
  --submit_path submit-qwen3-4b-instruct-2507-int8.csv
```

### Notes

- Modify `code/config.py` to adjust model registry or defaults.
- When passing a raw HF model id to `--model_name`, it will default to 8-bit loading unless changed in code.
