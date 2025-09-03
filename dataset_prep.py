import os
from pathlib import Path
import shutil
import random

# Dataset paths
RAW_DIR = Path("data/Indian_bovine_breeds")   # manually unzipped dataset
BASE_DIR = Path("data/indian_breeds")         # cleaned dataset output dir

def prepare_dataset():
    # Remove old processed dataset
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError("❌ Dataset not found. Please unzip it inside data/Indian_bovine_breeds/")

    # Create train/val folders (we skip test since you want manual testing later)
    for split in ["train", "val"]:
        (BASE_DIR / split).mkdir(parents=True, exist_ok=True)

    # Split each breed folder into train/val
    random.seed(42)
    for breed_dir in RAW_DIR.iterdir():
        if breed_dir.is_dir():
            images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.png"))
            random.shuffle(images)

            n = len(images)
            train_split = images[: int(0.8 * n)]
            val_split   = images[int(0.8 * n):]

            # Copy into split folders
            for img in train_split:
                shutil.copy(img, BASE_DIR/"train"/f"{breed_dir.name}_{img.name}")
            for img in val_split:
                shutil.copy(img, BASE_DIR/"val"/f"{breed_dir.name}_{img.name}")

    print(f"✅ Dataset prepared in {BASE_DIR}")

if __name__ == "__main__":
    prepare_dataset()
