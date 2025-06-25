# FRAMES-VQA  
**Benchmarking Fine-Tuning Robustness Across Multi-Modal Shifts in Visual Question Answering**  
[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue.svg)](https://cvpr2025.thecvf.com/)  
[![arXiv](https://img.shields.io/badge/arXiv-2505.21755-b31b1b.svg)](https://arxiv.org/abs/2505.21755)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)  

> Official implementation of **CVPR 2025** paper: _“FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering”_  
> 📄 [Paper](https://arxiv.org/abs/2505.21755) | 🧠 [Project Page](https://github.com/chengyuehuang511/FRAMES-VQA)

---

## 🧭 Overview

**FRAMES-VQA** is a benchmark for evaluating the robustness of fine-tuned Vision-Language Models (VLMs) under various multi-modal distribution shifts in Visual Question Answering (VQA).

### 🔑 Key Features
- 📦 Supports **LLaVA-1.5-7B** & **PaliGemma-3B**
- 📊 Benchmarks across **10+ VQA datasets**
- 🧪 Fine-tuning via **LoRA**, **linear probing**, or **full FT**
- 🔍 Shift categories: uni-modal (vision/question/answer), multi-modal, adversarial, far-OOD

---

## ⚙️ Installation

### Requirements
- Python 3.9+, PyTorch 2.0+
- CUDA GPU (≥40GB recommended)

### Setup

```bash
git clone https://github.com/chengyuehuang511/FRAMES-VQA.git
cd FRAMES-VQA
conda create -n frames-vqa python=3.9
conda activate frames-vqa
pip install -r requirements.txt
source setup_env.sh
```

---

## 📁 Dataset Structure

```
datasets_src/
├── datasets_files/    # Processed annotation files
└── images/            # Image directories
```

Supported datasets: VQAv2, OK-VQA, TextVQA, VizWiz, GQA, GQA-OOD, AdvQA, VQA-CP, VQA-CE, Rephrasings, etc.

---

## 🚀 Quick Start

### ✅ Train
```bash
# LLaVA on VQAv2
python train.py --cfg-path projects/llava/vqav2_train.yaml
```

### 📈 Evaluate
```bash
# Evaluate on single dataset
python evaluate.py --cfg-path projects/llava/vqav2_val.yaml
```

### SLURM
```bash
bash sbatch.sh        # Training
bash sbatch_test.sh   # Evaluation
```

---

## 🧪 Fine-Tuning Strategies

| Strategy         | `use_lora` | `linear_probe` |
|------------------|------------|----------------|
| LoRA             | 1          | 0              |
| Linear Probe     | 0          | 1              |
| Full Fine-tuning | 0          | 0              |

Modify in your `*_train.yaml`. Also extend `opt` to support more robust fine-tuning optimizers: [AdamP](https://arxiv.org/abs/2310.19182) for **FTP** and [AdamH](https://arxiv.org/abs/2411.01713) for **SPD**.

---

## 🧠 Supported Models

| Model Name       |
|------------------|
| LLaVA-1.5-7B     |
| PaliGemma-3B     |

---

## 📊 Supported Datasets (Grouped by Shift Type)

| Shift Type          | Datasets                                |
|---------------------|-----------------------------------------|
| **In-Distribution** | VQAv2                                   |
| **Vision Shift**    | IV-VQA, CV-VQA                          |
| **Question Shift**  | VQA-Rephrasings                         |
| **Answer Shift**    | VQA-CP                                  |
| **Multi-Modal Shift** | VQA-CE                      |
| **Adversarial**     | AdvQA                                   |
| **Far-OOD**         | OK-VQA, TextVQA, VizWiz                                 |

---

## 🧬 Configuration

- `configs/models/`: Model architecture definitions
- `configs/datasets/`: Dataset-level config
- `projects/`: Train/test/val experiment files

Naming:
- `*_train.yaml`: Training
- `*_test.yaml`: Evaluation
- `*_val.yaml`: Validation

---

## 🔁 Reproducibility

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

- Fixed seeds
- Deterministic CUDA
- Consistent dataloaders

---

## 📝 Citation

```bibtex
@inproceedings{huang2025framesvqa,
  title={FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering},
  author={Chengyue Huang and Brisa Maneechotesuwan and Shivang Chopra and Zsolt Kira},
  booktitle={CVPR},
  year={2025},
  url={https://arxiv.org/abs/2505.21755}
}
```

---

## 🙏 Acknowledgments

Built using [LAVIS](https://github.com/salesforce/LAVIS) and 🤗 Hugging Face Transformers.

---

> Active research code. Some features may be experimental.