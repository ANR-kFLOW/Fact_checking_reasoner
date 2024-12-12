import torch
from torch.utils.data import Dataset
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load RoBERTa model
roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
roberta_model.to(device)

# Custom Dataset Class
class EventRelationDataset(Dataset):
    def __init__(self, event1_embeddings, event2_embeddings, labels):
        self.event1_embeddings = event1_embeddings.cpu()
        self.event2_embeddings = event2_embeddings.cpu()
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        concatenated_embeddings = torch.cat((self.event1_embeddings[idx], self.event2_embeddings[idx]), dim=-1)
        seq_length = 1  # Single sequence representation
        attention_mask = torch.ones((1, seq_length))  # Attention mask for the sequence
        return {
            'inputs_embeds': concatenated_embeddings.unsqueeze(0),  # Add batch dimension
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Data Collator
def custom_data_collator(batch):
    inputs_embeds = pad_sequence([item['inputs_embeds'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Preprocess Data
def preprocess_data(df):
    event1_texts = df['sub'].tolist()
    event2_texts = df['obj'].tolist()
    relations = df['relation'].apply(lambda x: {'cause': 0, 'intend': 1, 'prevent': 2, 'enable': 3, 'no_relation': 4}[x]).tolist()

    # Generate SBERT embeddings
    event1_embeddings = sbert_model.encode(event1_texts, convert_to_tensor=True)
    event2_embeddings = sbert_model.encode(event2_texts, convert_to_tensor=True)

    return event1_embeddings, event2_embeddings, relations

# Load data
data_path = '/data/Youss/RE/new_data/cs_splited/cleaned_duplicates/transformed/'
train_data = pd.read_csv(data_path + 'train.csv').head(3)
dev_data = pd.read_csv(data_path + 'dev.csv').head(3)
test_data = pd.read_csv(data_path + 'test.csv').head(3)

train_event1_embeddings, train_event2_embeddings, train_labels = preprocess_data(train_data)
dev_event1_embeddings, dev_event2_embeddings, dev_labels = preprocess_data(dev_data)
test_event1_embeddings, test_event2_embeddings, test_labels = preprocess_data(test_data)

# Create datasets
train_dataset = EventRelationDataset(train_event1_embeddings, train_event2_embeddings, train_labels)
dev_dataset = EventRelationDataset(dev_event1_embeddings, dev_event2_embeddings, dev_labels)
test_dataset = EventRelationDataset(test_event1_embeddings, test_event2_embeddings, test_labels)

import torch.nn as nn


class EventRelationClassifier(nn.Module):
    def __init__(self, roberta_model):
        super(EventRelationClassifier, self).__init__()
        self.roberta = roberta_model
        self.projection_layer = nn.Linear(2 * 384, 768)  # Project to RoBERTa input size
        self.classifier = nn.Linear(768, 5)  # Output layer for classification
        self.loss_fn = nn.CrossEntropyLoss()  # Define the loss function

    def forward(self, inputs_embeds, attention_mask, labels=None):
        # Project SBERT embeddings to RoBERTa input size
        batch_size = inputs_embeds.size(0)
        seq_length = 1
        hidden_size = inputs_embeds.size(-1)
        projected_embeddings = inputs_embeds.view(batch_size, seq_length, hidden_size)

        # Pass to the RoBERTa model
        outputs = self.roberta(inputs_embeds=projected_embeddings, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, save_directory, roberta_model):
        model = cls(roberta_model)
        model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin")))
        print(f"Model loaded from {save_directory}")
        return model


# Initialize Model
model = EventRelationClassifier(roberta_model)
model.to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=custom_data_collator,
)

# Train the Model
trainer.train()

# Evaluate the Model
trainer.evaluate(eval_dataset=test_dataset)

# Save the Model
model.save_pretrained("./model")
