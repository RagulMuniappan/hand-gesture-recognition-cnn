import os
import random

# 🔹 CHANGE THIS PATH
FOLDER_PATH = r"C:/project_source_code/hand-gesture-recognition-cnn/dataset/test/9"

# Number of images to keep
KEEP_COUNT = 250

# Supported image extensions
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Get all images
images = [
    f for f in os.listdir(FOLDER_PATH)
    if f.lower().endswith(IMAGE_EXTS)
]

# Safety check
if len(images) < KEEP_COUNT:
    raise ValueError("Not enough images in the folder!")

# Randomly select images to keep
keep_images = set(random.sample(images, KEEP_COUNT))

# Delete the rest
for img in images:
    if img not in keep_images:
        os.remove(os.path.join(FOLDER_PATH, img))

print(f"✅ Kept {KEEP_COUNT} images, deleted {len(images) - KEEP_COUNT}")
