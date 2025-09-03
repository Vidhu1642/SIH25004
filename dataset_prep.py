import os
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import random
import json

# ✅ Load Kaggle credentials directly from kaggle.json file in project folder
KAGGLE_JSON = Path("kaggle.json")
if not KAGGLE_JSON.exists():
    raise FileNotFoundError("❌ kaggle.json not found in project folder. Please add it.")

with open(KAGGLE_JSON, "r") as f:
    creds = json.load(f)
os.environ["KAGGLE_USERNAME"] = creds["username"]
os.environ["KAGGLE_KEY"] = creds["key"]

# Dataset details
DATASET_NAME = "lukex9442/indian-bovine-breeds"
RAW_DIR = Path("data/Indian Bovine Breeds")   # name used inside the Kaggle zip
BASE_DIR = Path("data/indian_breeds")         # cleaned dataset output dir

def download_dataset():
    if RAW_DIR.exists():
        print("✅ Dataset already exists, skipping download.")
        return
    print("⬇️ Downloading dataset from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET_NAME,
        "-p", "data/", "--unzip"
    ], check=True)
    print("✅ Dataset downloaded and unzipped.")

def prepare_dataset():
    """Prepare dataset with train/val/test split."""
    # Clean old processed dataset
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)

    # Download if not present
    download_dataset()

    if not RAW_DIR.exists():
        raise FileNotFoundError("❌ Dataset not found after download. Check Kaggle setup.")

    # Create split folders
    for split in ["train", "val", "test"]:
        (BASE_DIR / split).mkdir(parents=True, exist_ok=True)

    # Split per breed
    random.seed(42)
    for breed_dir in RAW_DIR.iterdir():
        if breed_dir.is_dir():
            images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.png"))
            if not images:
                continue
            random.shuffle(images)

            n = len(images)
            train_split = images[: int(0.8 * n)]
            val_split   = images[int(0.8 * n): int(0.9 * n)]
            test_split  = images[int(0.9 * n):]

            for img in train_split:
                shutil.copy(img, BASE_DIR / "train" / f"{breed_dir.name}_{img.name}")
            for img in val_split:
                shutil.copy(img, BASE_DIR / "val" / f"{breed_dir.name}_{img.name}")
            for img in test_split:
                shutil.copy(img, BASE_DIR / "test" / f"{breed_dir.name}_{img.name}")

    print(f"✅ Dataset prepared at {BASE_DIR.resolve()}")

if __name__ == "__main__":
    prepare_dataset()
