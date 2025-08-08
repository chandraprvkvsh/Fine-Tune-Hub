import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
)
import wandb

# Authenticate (replace tokens with env vars or config for production)
os.system('huggingface-cli login --token "<your_hf_token>"')
wandb.login(key="<your_wandb_key>")
run = wandb.init(project="<your_wandb_project_name>", job_type="training", anonymous="allow")

# Load and preprocess data
df = pd.read_csv('Dataframe_Papers_V3.csv', usecols=['title', 'cite_read_boost'])
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
print(f"Threshold for clickbait classification: {threshold}")

data = df_final.sample(frac=1).reset_index(drop=True)
data.loc[:, 'clickbait'] = (data['cite_read_boost'] > threshold).astype(int)
data = data.drop(['cite_read_boost'], axis=1)

# Split data
train_df, val_df = train_test_split(data, test_size=0.01, random_state=42)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
train_inputs = tokenizer(train_df['title'].tolist(), padding=True, truncation=True, return_tensors="pt")
val_inputs = tokenizer(val_df['title'].tolist(), padding=True, truncation=True, return_tensors="pt")

train_labels = torch.tensor(train_df['clickbait'].tolist())
val_labels = torch.tensor(val_df['clickbait'].tolist())

train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
val_dataset = torch.utils.data.TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'google/flan-t5-base',
    num_labels=2,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Training setup
training_args = TrainingArguments(
    output_dir="./flan_t5_outputs",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=25,
    save_strategy="no",
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    report_to="wandb",
    load_best_model_at_end=True
)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"precision": precision, "recall": recall, "f1": f1}

# Collator for batching
class CustomDataCollator:
    def __call__(self, features):
        input_ids = [f[0] for f in features]
        attention_mask = [f[1] for f in features]
        labels = [f[2] for f in features]
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.tensor(labels)
        }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=CustomDataCollator(),
    tokenizer=tokenizer,
)

# Train
trainer.train()
wandb.finish()

# Push to Hub (optional)
# trainer.model.push_to_hub("<your_huggingface_repo>")
# tokenizer.push_to_hub("<your_huggingface_repo>")
