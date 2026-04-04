import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import wandb
import os
import re
import numpy as np

# --- 1. Extraction Logic ---
def read_text_files_with_labels(path):
    texts, labels = [], []
    # Filter for directories and ignore hidden folders like .ipynb_checkpoints
    class_folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and not f.startswith('.')])
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if os.path.isfile(file_path):
                file_name_no_ext, _ = os.path.splitext(file_name)
                # Clean text: replace underscores and remove digits
                text = file_name_no_ext.replace('_', ' ')
                text_without_digits = re.sub(r'\d+', '', text)
                texts.append(text_without_digits)
                labels.append(label_map[class_name])

    return np.array(texts), np.array(labels), class_folders

# --- 2. Custom Dataset Class ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        # Use the standard __call__ and squeeze(0) to ensure consistent shape [max_len]
        encoding = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# --- 3. Initialize W&B ---
wandb.init(project="garbage-TL-text", config={
    "learning_rate": 2e-5,
    "epochs": 10,
    "batch_size": 64,
    "model_name": "distilbert-base-uncased",
    "max_length": 64
})
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 4. Load & Prepare Data ---
train_texts, train_labels, class_names = read_text_files_with_labels('/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Train/')
val_texts, val_labels, _ = read_text_files_with_labels('/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Val')
test_texts, test_labels, _ = read_text_files_with_labels('/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Test')

tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)

train_loader = DataLoader(TextDataset(train_texts, train_labels, tokenizer, config.max_length), batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(TextDataset(val_texts, val_labels, tokenizer, config.max_length), batch_size=config.batch_size)
test_loader = DataLoader(TextDataset(test_texts, test_labels, tokenizer, config.max_length), batch_size=config.batch_size)

# --- 5. Model Setup ---
model = DistilBertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=len(class_names)
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
best_val_acc = 0.0

# --- 6. Training & Validation ---
for epoch in range(config.epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)

        outputs = model(ids, attention_mask=mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_acc = correct / total
    wandb.log({"epoch": epoch + 1, "train_loss": total_train_loss/len(train_loader), "val_acc": val_acc})
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_bert_model.pth")
        print(f"Epoch {epoch+1}: New best Val Acc {val_acc:.4f} - Saved.")

# --- 7. Final Test Phase ---
print("\n--- Final Evaluation on Test Set ---")
model.load_state_dict(torch.load("best_bert_model.pth"))
model.eval()

test_correct, test_total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)
        
        outputs = model(ids, attention_mask=mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        test_correct += (preds == targets).sum().item()
        test_total += targets.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

final_test_acc = test_correct / test_total

# Log final metrics and Confusion Matrix
wandb.log({
    "final_test_accuracy": final_test_acc,
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels, 
        preds=all_preds,
        class_names=class_names
    )
})

print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
wandb.finish()
