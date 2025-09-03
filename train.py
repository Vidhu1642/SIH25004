import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import os

# Paths
DATA_DIR = Path("data/indian_breeds")

# Custom Dataset that infers class from filename prefix
class FlatFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.transform = transform
        self.samples = []
        self.classes = []

        for img_file in self.folder.glob("*.jpg"):
            label = img_file.name.split("_")[0]  # e.g., "Kankrej_image1.jpg" → "Kankrej"
            if label not in self.classes:
                self.classes.append(label)
            self.samples.append((img_file, label))

        # Map classes to indices
        self.class_to_idx = {c: i for i, c in enumerate(sorted(self.classes))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = FlatFolderDataset(DATA_DIR / "train", transform=transform)
val_dataset   = FlatFolderDataset(DATA_DIR / "val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

num_classes = len(train_dataset.classes)
print("✅ Classes:", train_dataset.classes)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {acc:.2%}")

# Save model + label map
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/breed_model.pth")
with open("models/labels.txt", "w") as f:
    for cls in train_dataset.classes:
        f.write(cls + "\n")

print("✅ Training complete. Model saved to models/breed_model.pth")
