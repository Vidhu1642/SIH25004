# pipeline.py
import torch, json
from torchvision import models, transforms
from torch import nn
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load labels
with open("models/label_map.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# Load model
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(id2label))
model.load_state_dict(torch.load("models/indian_breeds.pth", map_location=DEVICE))
model = model.to(DEVICE).eval()

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred = probs.argmax()
    return id2label[pred], float(probs[pred])
