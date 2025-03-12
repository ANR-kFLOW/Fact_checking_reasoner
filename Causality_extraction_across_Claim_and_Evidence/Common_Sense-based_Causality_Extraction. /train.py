import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_path = '/data/Youss/RE/new_data/cs_splited/cleaned_duplicates/transformed/'
train_data = pd.read_csv(data_path + 'train.csv')
dev_data = pd.read_csv(data_path + 'dev.csv')
test_data = pd.read_csv(data_path + 'test.csv')
train_data = train_data.astype(str)
dev_data = dev_data.astype(str)
test_data = test_data.astype(str)
train_class_counts = train_data['relation'].value_counts()
dev_class_counts = dev_data['relation'].value_counts()
test_class_counts = test_data['relation'].value_counts()


print("Train class counts:")
print(train_class_counts)

print("\nDev class counts:")
print(dev_class_counts)

print("\nTest class counts:")
print(test_class_counts)

def process_data(df):
    event1_texts = df['sub'].tolist()
    event2_texts = df['obj'].tolist()
    relations = df['relation'].apply(lambda x: {'cause': 0, 'intend': 1, 'prevent': 2, 'enable': 3, 'no_relation': 4}[x]).tolist()

    return event1_texts, event2_texts, relations

train_event1, train_event2, train_relations = process_data(train_data)
dev_event1, dev_event2, dev_relations = process_data(dev_data)
test_event1, test_event2, test_relations = process_data(test_data)

train_inputs = [e1 + " [SEP] " + e2 for e1, e2 in zip(train_event1, train_event2)]
dev_inputs = [e1 + " [SEP] " + e2 for e1, e2 in zip(dev_event1, dev_event2)]
test_inputs = [e1 + " [SEP] " + e2 for e1, e2 in zip(test_event1, test_event2)]

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(inputs):
    return tokenizer(inputs, padding=True, truncation=True, max_length=128)

train_encodings = tokenizer(train_inputs, padding=True, truncation=True, max_length=128)
dev_encodings = tokenizer(dev_inputs, padding=True, truncation=True, max_length=128)
test_encodings = tokenizer(test_inputs, padding=True, truncation=True, max_length=128)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_relations
})
dev_dataset = Dataset.from_dict({
    'input_ids': dev_encodings['input_ids'],
    'attention_mask': dev_encodings['attention_mask'],
    'labels': dev_relations
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_relations
})

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)

model.to(device)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1',

    no_cuda=not torch.cuda.is_available(),
)

epoch_metrics = []

def compute_metrics(p):
    pred, labels = p
    pred = pred.argmax(axis=1)

    accuracy = accuracy_score(labels, pred)

    # Compute precision, recall, F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average=None, labels=[0, 1, 2, 3, 4])

    # Save metrics for each class
    for i, label in enumerate(['cause', 'intend', 'prevent', 'enable', 'no_relation']):
        print(f"Class '{label}' - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1[i]:.4f}")

    # Return the weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(labels, pred,
                                                                                          average='weighted')

    # Save metrics for this epoch
    epoch_metrics.append({
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1
    })

    return {'accuracy': accuracy, 'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1}


def plot_metrics(metrics, output_dir='./results'):
    # Plot the metrics after training
    metrics_df = pd.DataFrame(metrics)
    epochs = range(len(metrics_df))

    # Plot Accuracy, Precision, Recall, F1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_df['accuracy'], label='Accuracy', color='blue')
    plt.plot(epochs, metrics_df['precision'], label='Precision', color='green')
    plt.plot(epochs, metrics_df['recall'], label='Recall', color='orange')
    plt.plot(epochs, metrics_df['f1'], label='F1 Score', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Metrics during Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.close()

    # Save metrics to a CSV file
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate(test_dataset)

model.save_pretrained('./best_model')
tokenizer.save_pretrained('./best_model')
plot_metrics(epoch_metrics, output_dir='./results')