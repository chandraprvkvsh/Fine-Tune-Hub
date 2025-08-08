import os
import re
import torch
import wandb
import warnings
import subprocess
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    set_seed
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType
)

# Install dependencies
deps = [
    "bitsandbytes==0.42.0",
    "transformers==4.38.2",
    "peft==0.9.0",
    "accelerate==0.27.2",
    "trl==0.7.11",
    "datasets==2.16.0",
    "wandb"
]
for dep in deps:
    subprocess.run(["pip", "install", "--force-reinstall", dep])

# Configs
MODEL_ID = "google/gemma-2b"
SEED = 123
OUTPUT_DIR = "/"
NUM_CLASSES = 2
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
EPOCHS = 3

set_seed(SEED)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load data
df = pd.read_csv("FinalDataset.csv", usecols=["title", "read_count"])
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["title"] = df["title"].apply(lambda t: re.sub(r'<.*?>', '', t[2:-2].lower()))
df["title"] = df["title"].str.replace("  ", "", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df["read_count"] > 0]

# Normalize virality
df["log_read_count"] = np.log(df["read_count"] + 1)
df["virality"] = MinMaxScaler().fit_transform(df["log_read_count"].values.reshape(-1, 1))
df.drop(columns=["read_count", "log_read_count"], inplace=True)

# Labeling
threshold = df["virality"].median()
df["clickbait"] = (df["virality"] > threshold).astype(int)
df.drop(columns=["virality"], inplace=True)

# Balance dataset
cb_0 = df[df["clickbait"] == 0].sample(n=100000, random_state=42)
cb_1 = df[df["clickbait"] == 1].sample(n=100000, random_state=42)
df = pd.concat([cb_0, cb_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split
train_df, test_df = train_test_split(df, test_size=0.01, stratify=df["clickbait"], random_state=42)
train_df.rename(columns={"clickbait": "label"}, inplace=True)
test_df.rename(columns={"clickbait": "label"}, inplace=True)

# Dataset preparation
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["title", "label"]]),
    "test": Dataset.from_pandas(test_df[["title", "label"]])
})

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenized_dataset = {
    split: dataset[split].map(lambda x: tokenizer(x["title"], truncation=True, max_length=64), batched=True)
    for split in dataset
}

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Load and prepare model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=64,
    lora_dropout=0.1,
    lora_alpha=32,
    target_modules="all-linear",
    task_type=TaskType.SEQ_CLS
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Init Weights & Biases
wandb.login(key="****")  # 🔒 REDACTED
wandb.init(project="Gemma-clickbait-classification")

# Metrics
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == labels).mean()}

# Training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=OUTPUT_DIR,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        max_grad_norm=0.3,
        report_to="wandb"
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
wandb.finish()

# Push to Hugging Face Hub
trainer.model.push_to_hub("/")
tokenizer.push_to_hub("/")
