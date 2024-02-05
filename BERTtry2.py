from transformers import BertTokenizer, BertForSequenceClassification, AdamW, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Assuming 'malayalam_train.tsv' is correctly formatted with 'text' and 'category' columns
df = pd.read_csv('malayalam_train.tsv', sep='\t', header=0)

# Map categories to integers
category_to_int = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Mixed_feelings': 3, 'unknown_state': 4}
df['category_int'] = df['category'].map(category_to_int)

# Tokenize and encode the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
max_length = 64

def encode_texts(texts):
    return tokenizer.batch_encode_plus(
        texts, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

# Prepare the dataset
class MalayalamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Encoding the data
encoded_data = encode_texts(df['text'].tolist())
labels = torch.tensor(df['category_int'].values)

# Splitting the dataset
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    encoded_data['input_ids'], labels, random_state=42, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(
    encoded_data['attention_mask'], labels, random_state=42, test_size=0.1)

train_dataset = MalayalamDataset({'input_ids': train_inputs, 'attention_mask': train_masks}, train_labels)
val_dataset = MalayalamDataset({'input_ids': validation_inputs, 'attention_mask': validation_masks}, validation_labels)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=5)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Define compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
