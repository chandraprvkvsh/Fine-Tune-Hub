import pip
import subprocess
import os
import torch
import wandb
import pandas as pd

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training, 
    get_peft_model
)
from datasets import Dataset
from trl import SFTTrainer

# Install required packages
subprocess.run(['pip', 'install', '--force-reinstall', 'bitsandbytes==0.42.0'])
subprocess.run(['pip', 'install', '--force-reinstall', 'transformers==4.38.2'])
subprocess.run(['pip', 'install', '--force-reinstall', 'peft==0.9.0'])
subprocess.run(['pip', 'install', '--force-reinstall', 'accelerate==0.27.2'])
subprocess.run(['pip', 'install', '--upgrade', 'trl'])
subprocess.run(['pip', 'install', '--force-reinstall', 'datasets==2.16.0'])
pip.main(['install', 'wandb'])

# Login
os.system('huggingface-cli login --token "<your_hf_token>"')
wandb.login(key="<your_wandb_key>")
run = wandb.init(project="<your_wandb_project>", job_type="training", anonymous="allow")

# Model and data
base_model = "mistralai/Mistral-7B-v0.1"
new_model = "<your_huggingface_repo>"

df = pd.read_csv('titlegendata.csv', usecols=['abstract', 'title'])
df = df.head(200000)

# Format prompts
def formatting_prompts_func(example):
    output_texts = []
    for i in range(df.shape[0]):
        text = (
            f"<s>[INST] Craft an intelligent, clear, insightful, and succinct one-line title "
            f"for the research paper, drawing inspiration from the abstract provided. \n"
            f"{example.iloc[i,0]} [/INST] {example.iloc[i,1]} </s>"
        )
        output_texts.append(text)
    return output_texts

dataset = formatting_prompts_func(df)

# Quantization
bnb_config = BitsAndBytesConfig(  
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_dropout=0.1,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(lora_config)
model = get_peft_model(model, peft_config)

# Training arguments
training_arguments = TrainingArguments(
    output_dir="<your_output_dir>",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)

# Prepare dataset
dataset2 = Dataset.from_dict({"text": dataset})

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset2,
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Train
trainer.train()

# Push to Hugging Face Hub (replace with your repo)
trainer.model.push_to_hub("<your_huggingface_repo>")
tokenizer.push_to_hub("<your_huggingface_repo>")

wandb.finish()
model.config.use_cache = True
model.eval()
