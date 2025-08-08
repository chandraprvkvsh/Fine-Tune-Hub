import numpy as np
import pandas as pd
import re
import os
import torch
import warnings
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, set_seed,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset
import wandb
from kaggle_secrets import UserSecretsClient

# Login
user_secrets = UserSecretsClient()
secret_hf = user_secrets.get_secret("<your_hf_secret_key>")
secret_wandb = user_secrets.get_secret("<your_wandb_secret_key>")

os.system(f"huggingface-cli login --token {secret_hf}")
wandb.login(key=secret_wandb)
run = wandb.init(project="<your_wandb_project>", job_type="training", anonymous="allow")

# Load and preprocess data
df = pd.read_csv('/kaggle/input/readsmodel/Dataframe_Papers_V3.csv', usecols=['title', 'cite_read_boost'])
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

df_sorted = df.sort_values(by='cite_read_boost', ascending=False)
df_final = df_sorted[df_sorted['cite_read_boost'] > 0.01]

threshold = df_final['cite_read_boost'].median()
print(f"Threshold (median cite_read_boost): {threshold}")

data = df_final.sample(frac=1).reset_index(drop=True)
data.loc[:, 'clickbait'] = (data['cite_read_boost'] > threshold).astype(int)
data = data.drop(['cite_read_boost'], axis=1)
data = data[:1000]

# Split data
traindf, testdf = train_test_split(data, test_size=0.1)

# Setup
SEED = 123
set_seed(SEED)
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DIR = '/kaggle/working/'
NUM_CLASSES = 2
EPOCHS, R, LORA_ALPHA, LORA_DROPOUT = 5, 64, 32, 0.1
BATCH_SIZE = 8
MODEL_ID = '<your_model_path_or_id>'

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(traindf[['title', 'clickbait']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(testdf[['title', 'clickbait']].reset_index(drop=True))
dataset = {'train': train_dataset, 'test': test_dataset}

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(tokenizer.padding_side, tokenizer.pad_token)

tokenized_dataset = {
    split: dataset[split].map(
        lambda x: tokenizer(x["title"], truncation=True, max_length=128),
        batched=True
    )
    for split in dataset
}

# Quantization config
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
    device_map="auto",
)
print(f"Pad token ID: {model.config.pad_token_id}")
model.config.use_cache = False

# Apply LoRA
lora_config = LoraConfig(
    r=R,
    lora_dropout=LORA_DROPOUT,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.SEQ_CLS,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Collator
data_collator = DataCollatorWithPadding(tokenizer)

# Training args
training_args = TrainingArguments(
    output_dir=DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=f"{DIR}/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="wandb",
    save_total_limit=2,
    seed=SEED,
    push_to_hub=False,
    fp16=True,
)

# Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model
trainer.save_model(DIR + "/final_model")
