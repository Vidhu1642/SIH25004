# dataset_prep.py
import os, shutil, random, zipfile, subprocess
from pathlib import Path

DATASET_NAME = "lukex9442/indian-bovine-breeds"
BASE_DIR = Path("data/indian_breeds")

def download_dataset():
    print("📥 Downloading dataset from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download", "-d", DATASET_NAME, "-p", "data/", "--unzip"
    ], check=True)

def prepare_dataset():
    # Download
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    download_dataset()

    raw_dir = Path("data/Indian Bovine Breeds")
    if not raw_dir.exists():
        raise FileNotFoundError("Dataset not found after download. Check Kaggle setup.")

    # Create train/val/test folders
    for split in ["train", "val", "test"]:
        (BASE_DIR / split).mkdir(parents=True, exist_ok=True)

    # Split each breed into train/val/test
    random.seed(42)
    for breed_dir in raw_dir.iterdir():
        if breed_dir.is_dir():
            images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.png"))
            random.shuffle(images)

            n = len(images)
            train_split = images[: int(0.8 * n)]
            val_split   = images[int(0.8 * n): int(0.9 * n)]
            test_split  = images[int(0.9 * n):]

            for img in train_split:
                shutil.copy(img, BASE_DIR/"train"/img.name)
            for img in val_split:
                shutil.copy(img, BASE_DIR/"val"/img.name)
            for img in test_split:
                shutil.copy(img, BASE_DIR/"test"/img.name)

    print(f"✅ Dataset prepared in {BASE_DIR}")

if __name__ == "__main__":
    prepare_dataset()
