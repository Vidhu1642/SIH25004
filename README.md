# 🐄 Indian Cattle & Buffalo Breed Recognition

A deep learning pipeline for recognizing **Indian cattle and buffalo breeds** from images using **PyTorch** and the Kaggle **Indian Bovine Breeds** dataset.

---

## 📂 Project Structure

indian_breed_recognition/
│
├── data/ # dataset will be automatically downloaded and unzipped here
├── models/ # trained model weights and label map
├── test_images/ # place a test image here as test.jpg
│
├── dataset_prep.py # downloads and prepares the dataset
├── train.py # trains the breed recognition model
├── pipeline.py # loads model and provides prediction function
├── test_model.py # simple script to test a fixed image
├── run_all.py # runs the whole pipeline (download → train → test)
│
├── requirements.txt # list of Python dependencies
└── README.md # this file



---

## 🐃 Breeds Included in Dataset

The dataset (`lukex9442/indian-bovine-breeds` on Kaggle) contains images for several Indian **cattle** and **buffalo** breeds.  
Some examples include:

- **Alambadi**
- **Amritmahal**
- **Ayrshire**
- **Banni**
- **Bargur**

(You can confirm the full list after running `dataset_prep.py` — each subfolder corresponds to a breed label.)

---

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
run_all.py → only installs requirements + prepares dataset + trains the model.
test_model.py → you run with your test.jpg