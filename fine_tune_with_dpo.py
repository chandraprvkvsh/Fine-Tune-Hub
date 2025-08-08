import os
import re
import subprocess
import torch
import wandb
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from trl import DPOTrainer

# === 1. Install dependencies ===
required_packages = [
    "bitsandbytes==0.42.0",
    "transformers==4.38.2",
    "peft==0.9.0",
    "accelerate==0.27.2",
    "trl",
    "datasets==2.16.0",
    "sentencepiece",
    "langchain-community",
    "langchain",
    "sentence-transformers",
    "faiss-gpu",
    "faiss-cpu",
    "huggingface_hub",
    "wandb"
]
for pkg in required_packages:
    subprocess.run(["pip", "install", "--upgrade", pkg])

# === 2. Login securely ===
os.system('huggingface-cli login --token "hf_****"')  # Replace with secure method
wandb.login(key="****")  # Replace with env variable or secret manager

# === 3. Initialize Weights & Biases ===
run = wandb.init(project="mistral_dpo", job_type="training", anonymous="allow")

# === 4. Load and preprocess dataset ===
dataset = load_dataset("/")["train"]
dataset = dataset.filter(lambda x: x['selected_title'] and x['rejected_title'] and x['abstract'])

# === 5. Vector DB for few-shot retrieval ===
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

db = FAISS.load_local("/", embed_model, allow_dangerous_deserialization=True)

def make_few_shot_db(docs):
    abstracts, titles = [], []
    for doc in docs[:2]:
        lines = doc.page_content.split('\n')
        if len(lines) > 4:
            titles.append(lines[2])
            abstracts.append(lines[4])
    return '\n\n'.join(abstracts + titles) + '\n\n'

# === 6. Tokenizer ===
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === 7. Format dataset to ChatML ===
def chatml_format(example):
    sim_docs = db.similarity_search(example['abstract'])
    context = make_few_shot_db(sim_docs)
    sys_msg = f"You are an experienced researcher... The papers you've written are:\n\n{context}"
    system = tokenizer.apply_chat_template([{"role": "system", "content": sys_msg}], tokenize=False)
    user_prompt = f"Write a witty and informative title based on the abstract below:\nAbstract: {example['abstract']}\n"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
    return {
        "prompt": system + prompt,
        "chosen": example['selected_title'] + "<im_end>\n",
        "rejected": example['rejected_title'] + "<im_end>\n"
    }

original_columns = dataset.column_names
train_dataset = dataset.map(chatml_format, remove_columns=original_columns)

# === 8. LoRA Config ===
peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
)

# === 9. Quantized Model Loading ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# === 10. Training Arguments ===
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    remove_unused_columns=False,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    save_strategy="no",
    logging_steps=100,
    output_dir="/",
    optim="paged_adamw_32bit",
    bf16=False,
    fp16=True,
    warmup_steps=50,
    report_to="wandb"
)

# === 11. DPO Training ===
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_length=1536
)

trainer.train()

# === 12. Push to Hugging Face Hub ===
trainer.model.push_to_hub("/")
tokenizer.push_to_hub("/")

wandb.finish()
