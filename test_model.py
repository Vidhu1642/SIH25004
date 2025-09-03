import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Paths
MODEL_PATH = "models/breed_model.pth"
LABELS_PATH = "models/labels.txt"
TEST_IMAGE = Path("test_images/test.jpg")  # put your test image here

# Load labels
with open(LABELS_PATH, "r") as f:
    classes = [line.strip() for line in f]

# Model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image
if not TEST_IMAGE.exists():
    raise FileNotFoundError(f"❌ Test image not found at {TEST_IMAGE}. Please add one.")

img = Image.open(TEST_IMAGE).convert("RGB")
img = transform(img).unsqueeze(0)  # add batch dimension

# Predict
with torch.no_grad():
    outputs = model(img)
    _, pred = torch.max(outputs, 1)
    predicted_class = classes[pred.item()]

print(f"✅ Prediction: {predicted_class}")

