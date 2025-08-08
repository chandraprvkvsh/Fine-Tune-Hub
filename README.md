# FineTune Hub
This repository provides a collection of ready-to-run Python scripts for fine-tuning any large language models. The scripts cover:

- Parameter-Efficient Fine-Tuning (PEFT) of large language models with techniques like LoRA and quantization for efficient training.
- Direct Preference Optimization (DPO) training for alignment using preference data.
- Reinforcement Learning from Human Feedback (RLHF) implementations with reward models based on LSTM or Transformer architectures.
- Serving fine-tuned models via FastAPI, including easy deployment using ngrok tunnels.

All scripts are designed to be generic, portable, and easily adaptable to different environments and projects. They include example setups for model loading, training arguments, dataset preparation, and pushing models to HuggingFace Hub. Minimal modifications are needed to customize tokens, paths, and project names.
