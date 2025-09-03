import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path

MODEL_PATH = Path("models/indian_breeds.pth")
LABEL_MAP_PATH = Path("models/label_map.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing for test images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    """Load trained model and label map."""
    with open(LABEL_MAP_PATH, "r") as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)

    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, idx_to_class

def predict_image(image_path: str):
    """Predict breed from an image path."""
    model, idx_to_class = load_model()

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_class = torch.max(probs, 1)

    breed_name = idx_to_class[str(pred_class.item())]
    confidence = conf.item() * 100
    return breed_name, confidence
