# test_model.py
from pipeline import predict_image

# Fixed image
img_path = "test_images/test.jpg"

label, conf = predict_image(img_path)
print(f"Prediction: {label} (confidence {conf:.2%})")
