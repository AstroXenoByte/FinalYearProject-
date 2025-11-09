import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Config
DATASET_DIR = "groundnut_dataset_split"
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 2e-5
ACCUMULATION_STEPS = 4  # For gradient accumulation

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Load datasets
train_data = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=transform)
val_data   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=transform)
test_data  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# Number of classes
num_classes = len(train_data.classes)
print("Classes detected:", train_data.classes)

# Load model (small and public)
model = ViTForImageClassification.from_pretrained(
    "facebook/deit-tiny-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

# Optimizer and loss
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    optimizer.zero_grad()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for step, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).logits
        loss = criterion(outputs, labels) / ACCUMULATION_STEPS  # normalize for accumulation
        loss.backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)

    train_acc = correct / len(train_data)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(val_data)
    print(f"Validation Accuracy: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "vit_groundnut_cpu_friendly.pth")
print(" Model saved: vit_groundnut_cpu_friendly.pth")
