# Vietnamese Hallucination Detection

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
|-- data/               # Training and test CSV files
|-- results/            # Inference outputs and ensembles
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
| | |-- README.md
| |
| |-- qwen3-4b/
| | |-- finetune.py
| | |-- inference.py
| |
| |-- qwen25-7b/
| |-- finetune.py
| |-- inference.py
|
|-- requirements.txt
|-- run_reference_and_ensemble.sh
|-- README.md
```

---

## 3. Checkpoints

Download all model checkpoints from the following link and place them in the `checkpoints/` directory: [**🗂️ Checkpoints (Google Drive)**](https://drive.google.com/drive/u/4/folders/16LOiPlNrREaae_moJL0dNZipGPTmLboJ)


## 4. Environment Setup

We recommend using a **virtual environment** or **conda environment**.
Make sure your PyTorch version is compatible with your local CUDA version.

```bash
python -m venv .venv
source .venv/bin/activate        # On Linux / macOS
pip install -U pip
pip install -r requirements.txt
```

## 5. Fine-tuning and Inference
Each model directory (under ```src/models/```) contains two main scripts:

- finetune.py — for supervised fine-tuning

- inference.py — for running predictions on test data

These scripts accept arguments such as ```--config``` (path to the configuration file) and ```--model_name```.

a) LLaMA Models (src/models/llama)

- Fine-tuning

```powershell
python .\src\models\llama\finetune.py \
        --config .\src\configs\config_llama.py \
        --model_name your_model_name \
        --mode train \
        --output_dir .\checkpoints\your_model_name
```

- Inference

```powershell
python .\src\models\llama\inference.py \
        --config .\src\configs\config_llama.py \
        --model_name your_model_name \
        --test_csv .\data\example_test.csv \
        --submit_path .\results\submit-model-name.csv
```

b) Qwen2.5-7B (`src/models/qwen25-7b`)
- Finetune
```powershell
python .\src\models\qwen25-7b\finetune.py
```
- Inference
```powershell
python .\src\models\qwen25-7b\inference.py
```

c) Qwen3-4B (`src/models/qwen3-4b`)

- Finetune
```powershell


```
- Inference
```powershell


```

## 6. Ensemble Strategy

To improve prediction robustness, we adopt a **two-stage ensemble pipeline** that combines outputs from multiple fine-tuned models.

### Stage 1: Model-wise Ensemble

Each model family (LLaMA or Qwen) is ensembled internally to produce a more stable prediction file.

- 🦙 **LLaMA Ensemble**
  - Combine predictions from all LLaMA variants (e.g., `Llama3.2-3B-4bit`, `Llama3.2-3B-16bit`, `Llama2-7B-16bit`)
  - Output: `results/llama_ensemble.csv`

- 🧬 **Qwen3 Ensemble**
  - Combine predictions from all Qwen3 models (e.g., `Qwen3-4B`, `Qwen3-4B-Instruct`)
  - Output: `results/qwen3_ensemble.csv`

- ⚡ **Qwen2.5 Model**
  - Use one single result file from the Qwen2.5 model
  - Output: `results/qwen25_result.csv`

Each ensemble file is generated using **majority voting**.


### Stage 2: Final Ensemble

At this level, we combine the first-stage results from all model families to form the final output. The final ensemble also applies **majority voting** to determine the most reliable label per sample.

---

## 🚀 7. Inference all-in-one

We provide a single shell script that automates inference and ensemble generation.

```bash
bash run_reference_and_ensemble.sh
```
