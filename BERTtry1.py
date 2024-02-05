import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Load the dataset
df = pd.read_csv('malayalam_train.tsv', sep='\t', header=0)

# Preprocess the data (as needed)

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['category'], test_size=0.2)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize the text
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# Convert to torch dataset
class MalayalamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # Convert labels to integers if they are not already
        self.labels = torch.tensor(labels.astype('int'))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]  # Labels are already tensors
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = MalayalamDataset(train_encodings, train_labels)
val_dataset = MalayalamDataset(val_encodings, val_labels)

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    # Remove or replace 'evaluate_during_training' if it's causing an error
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()
