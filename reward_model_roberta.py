# Install required packages
import subprocess
subprocess.run(['pip', 'install', '--upgrade', 'torch'])
subprocess.run(['pip', 'install', '--upgrade', 'transformers'])
subprocess.run(['pip', 'install', '--upgrade', 'accelerate'])

# Authenticate with Hugging Face
import os
os.system('huggingface-cli login --token "<your_hf_token>"')

# Initialize Weights & Biases
import wandb
wandb.login(key="<your_wandb_key>")
wandb.init(project="<your_wandb_project_name>")

# Load and clean data
import pandas as pd
import re
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

# Filter and label data
df_sorted = df.sort_values(by='cite_read_boost', ascending=False)
df_final = df_sorted[df_sorted['cite_read_boost'] > 0.01]
threshold = df_final['cite_read_boost'].median()
data = df_final.sample(frac=1).reset_index(drop=True)
data.loc[:, 'clickbait'] = (data['cite_read_boost'] > threshold).astype(int)
data = data.drop(['cite_read_boost'], axis=1)

# Prepare datasets
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

class ClickbaitDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        title = str(self.data.iloc[idx]['title'])
        label = self.data.iloc[idx]['clickbait']
        encoding = self.tokenizer(title, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, torch_dtype=torch.bfloat16, device_map='auto')

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    evaluation_strategy="steps",
    eval_steps=250,
    logging_dir='./logs',
    logging_steps=50,
    report_to="wandb"
)

train_dataset = ClickbaitDataset(train_data, tokenizer)
val_dataset = ClickbaitDataset(val_data, tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
    }

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
wandb.finish()

# Push to Hugging Face Hub
trainer.model.push_to_hub("<your_huggingface_repo>")
tokenizer.push_to_hub("<your_huggingface_repo>")
