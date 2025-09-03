import zipfile
from pathlib import Path

zip_path = Path("data/indian-bovine-breeds.zip")
extract_dir = Path("data/")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Dataset extracted!")
