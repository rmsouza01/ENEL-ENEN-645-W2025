import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import wandb
import os

# --- 1. Initialize W&B ---
wandb.init(
    project="garbage-TL-image",
    config={
        "learning_rate": 0.001,
        "epochs": 20,
        "batch_size": 64,
        "architecture": "EfficientNetV2-m",
        "dataset": "CustomDataset"
    }
)
config = wandb.config

# --- 2. Setup Device & Paths ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
TRAIN_DIR = '/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Train/'
VAL_DIR = '/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Val'
TEST_DIR = '/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Test'

# --- 3. Data Augmentation and Transforms ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize(480),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transforms['train'])
val_dataset = datasets.ImageFolder(VAL_DIR, data_transforms['val_test'])
test_dataset = datasets.ImageFolder(TEST_DIR, data_transforms['val_test'])

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2,pin_memory=True,prefetch_factor=2,persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2,pin_memory=True,prefetch_factor=2,persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2,pin_memory=True,prefetch_factor=2,persistent_workers=True)

class_names = train_dataset.classes
num_classes = len(class_names)

# --- 4. Initialize MobileNetV2 ---
#model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

# --- 5. Loss, Optimizer, and Tracking ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=config.learning_rate)

wandb.watch(model, log="all")
best_val_acc = 0.0

# --- 6. Training & Validation Loop ---
for epoch in range(config.epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation Phase
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total

    # Log metrics
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc
    })

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_ef_v2.pth")
        print("--> Best model weights saved.")

# --- 7. Final Test Phase ---
print("\n--- Final Evaluation on Test Set ---")
model.load_state_dict(torch.load("best_model_ef_v2.pth")) # Load the best weights
model.eval()

test_loss, correct, total = 0.0, 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_acc = 100. * correct / total

# Log final test data and confusion matrix
wandb.log({
    "test_loss": avg_test_loss,
    "test_accuracy": test_acc,
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels, 
        preds=all_preds,
        class_names=class_names
    )
})

print(f"Final Test Accuracy: {test_acc:.2f}%")
wandb.finish()
