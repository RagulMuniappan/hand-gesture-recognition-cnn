import pickle
import matplotlib.pyplot as plt
import numpy as np

# 1. Load training history
try:
    with open('train_history.pkl', 'rb') as f:
        history = pickle.load(f)
    print("History loaded successfully!")
except FileNotFoundError:
    print("Error: train_history.pkl not found. Run train.py first!")
    exit()

# 2. Prepare data
epochs = np.arange(1, len(history['accuracy']) + 1)
bar_width = 0.35

# Accuracy Bar Graph
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(
    epochs - bar_width/2,
    history['accuracy'],
    bar_width,
    label='Train Accuracy'
)
plt.bar(
    epochs + bar_width/2,
    history['val_accuracy'],
    bar_width,
    label='Val Accuracy'
)

plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Loss Bar Graph
plt.subplot(1, 2, 2)
plt.bar(
    epochs - bar_width/2,
    history['loss'],
    bar_width,
    label='Train Loss'
)
plt.bar(
    epochs + bar_width/2,
    history['val_loss'],
    bar_width,
    label='Val Loss'
)

plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
