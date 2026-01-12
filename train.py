from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle

# 1. Model Definition
model = Sequential()
model.add(Input(shape=(256, 256, 1)))  # Grayscale input

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(11, activation='softmax'))  # 11 classes

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 2. Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

class_list = ['0','1','2','3','4','5','6','7','8','9','unknown']

training_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=class_list,
    class_mode='categorical'
)

val_set = val_datagen.flow_from_directory(
    'dataset/test',
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=class_list,
    class_mode='categorical'
)

# 3. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(
        "best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# 4. TRAINING (FIXED)
history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=5,
    validation_data=val_set,
    validation_steps=len(val_set),
    callbacks=callbacks
)

# 5. Save Final Model
model.save("final_hand_model.keras")
print("Training complete. Final model saved.")

# 6. Save Training History (FIXED)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(BASE_DIR, "train_history.pkl")

with open(history_path, "wb") as f:
    pickle.dump(history.history, f)

print("Training history saved to:", history_path)