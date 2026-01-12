from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os


# Load FULL model
model = load_model("final_hand_model.keras")
print("Loaded full model successfully")


# Class labels
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown']


# Classification function
def classify(img_path):
    test_image = image.load_img(
        img_path,
        target_size=(256, 256),
        color_mode="grayscale"
    )

    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    preds = model.predict(test_image, verbose=0)[0]

    idx = np.argmax(preds)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Prediction: {classes[idx]}")
    print(f"Confidence: {preds[idx]:.4f}")
    print("-" * 40)


# Run on folder
path = "check"

for file in os.listdir(path):
    full_path = os.path.join(path, file)
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        classify(full_path)