# train.py
import torch, copy, json
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 224, 32, 15, 3e-4
DATA_DIR = Path("data/indian_breeds")

def train_model():
    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Datasets
    train_ds = datasets.ImageFolder(DATA_DIR/"train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(DATA_DIR/"val", transform=val_tfms)
    test_ds  = datasets.ImageFolder(DATA_DIR/"test", transform=val_tfms)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    class_names = train_ds.classes
    print("Breeds:", class_names)

    # Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    def run_epoch(dl, train=False):
        if train: model.train()
        else: model.eval()
        loss_sum, correct, total = 0, 0, 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train: optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x)
                loss = criterion(out, y)
                if train:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return loss_sum/total, correct/total

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(train_dl, train=True)
        val_loss, val_acc = run_epoch(val_dl, train=False)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train acc {tr_acc:.3f} | Val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

    # Load best model
    model.load_state_dict(best_wts)

    # Test evaluation
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            out = model(x.to(DEVICE))
            preds = out.argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    # Save model + label map
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/indian_breeds.pth")
    with open("models/label_map.json", "w") as f:
        json.dump({i: c for i, c in enumerate(class_names)}, f)

    print("✅ Model saved to models/indian_breeds.pth")

if __name__ == "__main__":
    train_model()
