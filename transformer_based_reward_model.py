import numpy as np
import pandas as pd
import os
import subprocess
import torch
import re
import wandb

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    set_seed,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType
)

# Install dependencies
subprocess.run(['pip', 'install', '--force-reinstall', 'bitsandbytes==0.42.0'])
subprocess.run(['pip', 'install', '--force-reinstall', 'transformers==4.38.2'])
subprocess.run(['pip', 'install', '--force-reinstall', 'peft==0.9.0'])
subprocess.run(['pip', 'install', '--force-reinstall', 'accelerate==0.27.2'])
subprocess.run(['pip', 'install', '--force-reinstall', 'trl==0.7.11'])
subprocess.run(['pip', 'install', '--force-reinstall', 'datasets==2.16.0'])
subprocess.run(['pip', 'install', '--force-reinstall', 'wandb'])

# Auth (tokens redacted)
os.system('huggingface-cli login --token "hf_****"')
wandb.login(key="****")
wandb.init(project='gemmavirality', job_type="training", anonymous="allow")

# Load and preprocess data
df = pd.read_csv('FinalDataset.csv', usecols=['title', 'read_count'])
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

def process_paragraph(paragraph):
    paragraph = paragraph.lower()
    paragraph = re.sub(r'<.*?>', '', paragraph)
    paragraph = paragraph.replace('  ', '')
    paragraph = re.sub(r'\s+', ' ', paragraph).strip()
    return paragraph

df['title'] = [title[2:-2] for title in df['title']]
df['title'] = df['title'].apply(process_paragraph)

df = df[df['read_count'] > 0]
df['log_read_count'] = np.log(df['read_count'] + 1)
scaler = MinMaxScaler()
df['virality'] = scaler.fit_transform(df['log_read_count'].values.reshape(-1, 1))

df = df.drop(columns=['read_count', 'log_read_count'])
threshold = df['virality'].median()
df['clickbait'] = (df['virality'] > threshold).astype(int)
df = df.drop(['virality'], axis=1)

# Balance and shuffle
cb_0 = df[df['clickbait'] == 0].sample(n=100000, random_state=42)
cb_1 = df[df['clickbait'] == 1].sample(n=100000, random_state=42)
data = pd.concat([cb_0, cb_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Train/test split
X = data[['title']]
y = data['clickbait']
train_df, test_df = train_test_split(data, test_size=0.01, stratify=y, random_state=42)

# Constants
SEED = 123
set_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DIR = ''
NUM_CLASSES = 2
MODEL_ID = 'google/gemma-2b'

# Convert to datasets
train_df.rename(columns={'clickbait': 'label'}, inplace=True)
test_df.rename(columns={'clickbait': 'label'}, inplace=True)
train_dataset = Dataset.from_pandas(train_df[['title', 'label']])
test_dataset = Dataset.from_pandas(test_df[['title', 'label']])
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenized_dataset = {
    split: dataset[split].map(lambda x: tokenizer(x["title"], truncation=True, max_length=64), batched=True)
    for split in dataset
}

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.config.use_cache = False

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_dropout=0.1,
    lora_alpha=32,
    target_modules='all-linear',
    task_type=TaskType.SEQ_CLS
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

wandb.init(project="Gemma-clickbait-classification")

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

# Training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=DIR,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        max_grad_norm=0.3,
        report_to="wandb"
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
wandb.finish()

# Push to Hub (optional)
trainer.model.push_to_hub("")
tokenizer.push_to_hub("")
