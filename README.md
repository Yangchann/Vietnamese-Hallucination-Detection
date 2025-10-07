# DSC2025 - Vietnamese Hallucination Detection

## 1. Project Overview

This repository implements a **Vietnamese hallucination detection** project that classifies the consistency between a model’s generated response and its given context.
Each input sample consists of a `(context, prompt, response)` triple, and the model predicts one of the following labels:

- **`no`** — The response is fully consistent with the context
- **`intrinsic`** — The response contradicts or misinterprets the context
- **`extrinsic`** — The response adds information not present or not inferable from the context

We fine-tune multiple **Large Language Models (LLMs)** — including **LLaMA** and **Qwen** variants — and combine their predictions through a **staged ensemble** strategy to achieve robust and accurate hallucination detection.

---

## 2. Repository Structure
```
Vietnamese-Hallucination-Detection
|-- checkpoints/        # Place downloaded checkpoints here
|
|-- data/   
| |-- vihallu-train.csv
| |-- vihallu-private-test.csv
|
|-- src/
| |-- configs/
| | |-- config_llama.py
| | |-- config_qwen3_4b.py
| | |-- config_qwen25_7b.py
| |
| |-- models/
| |-- llama/
| | |-- finetune.py
| | |-- inference.py
| |
| |-- qwen3-4b/
| | |-- finetune.py
| | |-- inference.py
| |
| |-- qwen25-7b/
| | |-- finetune.py
| | |-- inference.py
|
|-- requirements.txt
|-- run_reference_and_ensemble.py
|-- README.md
```

---

## 3. System Requirements

To ensure stable training and inference, we recommend the following setup:

| Component | Minimum Requirement | Recommended Setup |
|------------|--------------------|-------------------|
| **Python** | 3.8+ | 3.10 |
| **CUDA GPU** | Required | ✅ NVIDIA GPU with CUDA support |
| **VRAM** | 16GB | 24GB+ (e.g., RTX 4090) |
| **RAM** | 16GB | 32GB |

---

## 4. Installation

```bash
git clone https://github.com/Yangchann/Vietnamese-Hallucination-Detection.git
cd Vietnamese-Hallucination-Detection
pip install -r requirements.txt
```

## 5. Checkpoints

Download all model checkpoints from the following link and place them in the `checkpoints/` directory: [**🗂️ Checkpoints (Google Drive)**](https://drive.google.com/drive/u/4/folders/16LOiPlNrREaae_moJL0dNZipGPTmLboJ)

```bash

gdown --folder https://drive.google.com/drive/u/4/folders/16LOiPlNrREaae_moJL0dNZipGPTmLboJ -O ./checkpoints

```

---

## 6. Fine-tuning and Inference
Each model directory (under ```src/models/```) contains two main scripts:

- finetune.py — for supervised fine-tuning

- inference.py — for running predictions on test data

These scripts accept arguments such as ```--config``` (path to the configuration file) and ```--model_name```.

a) LLaMA Models (src/models/llama)

- Fine-tuning

```powershell
python src/models/llama/finetune.py \
        --config src/configs/config_llama.py \
        --model_name your_model_name \
        --mode train \
        --output_dir checkpoints/your_model_name
```

- Inference

```powershell
python src/models/llama/inference.py \
        --config src/configs/config_llama.py \
        --model_name your_model_name \
        --test_csv data/vihallu-private-test.csv \
        --submit_path /results/submit-model-name.csv
```
Note: ```your_model_name``` including ```Llama32-3B-4bit```, L```lama32-3B-16bit```, and ```Llama2-7B-16bit```

b) Qwen3-4B (`src/models/qwen3-4b`)

- Finetune
```powershell
python src/models/qwen3-4b/finetune.py \
        --config src/configs/config_qwen3_4b.py \
        --model_name your_model_name \
        --mode train \
        --output_dir checkpoints/your_model_name
```
- Inference
```powershell
python src/models/qwen3-4b/inference.py \
        --config src/configs/config_qwen3_4b.py \
        --model_name your_model_name \
        --test_csv data/vihallu-private-test.csv \
        --submit_path submit-model-name.csv
```
Note: ```your_model_name``` including ```qwen3-4b-instruct-2507-int8```, ```qwen3-4b-instruct-2507-full```, and ```qwen3-4b-thinking-2507-int8```


c) Qwen2.5-7B (`src/models/qwen25-7b`)
- Finetune
```powershell
python src/models/qwen25-7b/finetune.py
```
- Inference
```powershell
python src/models/qwen25-7b/inference.py
```

---

## 7. Ensemble Strategy

To improve prediction robustness, we adopt a **two-stage ensemble pipeline** that combines outputs from multiple fine-tuned models.

### Stage 1: Model-wise Ensemble

Each model family (LLaMA or Qwen) is ensembled internally to produce a more stable prediction file.

- 🦙 **LLaMA Ensemble**
  - Combine predictions from all LLaMA variants (e.g., `Llama3.2-3B-4bit`, `Llama3.2-3B-16bit`, `Llama2-7B-16bit`)
  - Output: `submit-Llama-Ensemble.csv`

- 🧬 **Qwen3 Ensemble**
  - Combine predictions from all Qwen3 models (e.g., `Qwen3-4B`, `Qwen3-4B-Instruct`)
  - Output: `submit-Qwen3-Ensemble.csv`

- ⚡ **Qwen2.5 Model**
  - Use one single result file from the Qwen2.5 model
  - Output: `submit-qwen25-7b.csv`

Each ensemble file is generated using **majority voting**.


### Stage 2: Final Ensemble

At this level, we combine the first-stage results from all model families to form the final output. The final ensemble also applies **majority voting** to determine the most reliable label per sample.

---

## 🚀 8. Inference all-in-one

We provide a single shell script that automates inference and ensemble generation.

```bash
python run_reference_and_ensemble.py
```


**⚠️ Note: Due to non-deterministic sampling (```temperature``` > 0) and random tie-breaking during ensemble, final predictions may slightly differ across runs.**
