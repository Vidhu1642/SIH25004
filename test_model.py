from pipeline import predict_image
from pathlib import Path

TEST_IMAGE = Path("test_images/test.jpg")

if not TEST_IMAGE.exists():
    print("❌ Please place a test image at test_images/test.jpg")
else:
    breed, confidence = predict_image(str(TEST_IMAGE))
    print(f"Prediction: {breed} (confidence {confidence:.2f}%)")
