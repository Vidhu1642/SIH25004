# run_all.py
import os

print("STEP 1: Preparing dataset...")
os.system("python dataset_prep.py")

print("\nSTEP 2: Training model...")
os.system("python train.py")

print("\nSTEP 3: Testing with sample image...")
os.system("python test_model.py")
