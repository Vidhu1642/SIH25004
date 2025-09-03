import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Helper to run a shell command and check errors."""
    print(f"\n▶️ Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

def main():
    # 1. Install requirements
    if Path("requirements.txt").exists():
        print("📦 Installing requirements...")
        run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("⚠️ requirements.txt not found, skipping installation.")

    # # 2. Prepare dataset
    # print("📥 Preparing dataset...")
    # run_command([sys.executable, "dataset_prep.py"])

    # # 3. Train model
    # print("🏋️ Training model...")
    # run_command([sys.executable, "train.py"])

    print("\n✅ Training completed! Now you can test using:")
    print("   python test_model.py")

if __name__ == "__main__":
    main()
