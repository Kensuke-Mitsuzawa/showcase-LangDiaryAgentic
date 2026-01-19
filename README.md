# LangDiaryAgentic


# Prerequisites

- Python: 3.12+
- Package Manager: uv
- Hardware: NVIDIA GPU with CUDA support (Recommended: [e.g., 24GB+] VRAM). Confirmed to work with CUDA 12.9 on 2 V100S-PCIE-32GB GPUs.
- OS: Linux.


# Installation & Setup

```
1. Environment Setup
This project uses uv for fast dependency management.

# Install uv if not already installed
pip install uv

# Sync dependencies and create virtual environment (default: .venv)
uv sync

# Activate the environment
source .venv/bin/activate
```

##  API Keys & LLM Access

These descriptions below are applied in the followig cases,
1. want to use the LLMs that requires the accepting "terms" on the HuggingFace (such as Mistral or Llama),
2. want to change the storage path where the `transformers` package saves the model file.


Create an Access Token (Read permission) in your HF settings. Create a .env file in the project root:

```
HF_TOKEN='[your_huggingface_token_here]'

# Optional: Custom model cache location (default is ~/.cache)
# HF_HOME='/abs/path/to/large/storage'
```


# Web App

```
streamlit run ui/main.py
```
