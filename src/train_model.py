import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

def load_data():
    X, y = [], []
    labels = {"AM": 0, "FM": 1, "Noise": 2}
    for file in os.listdir("data/spectrograms"):
        img = Image.open(f"data/spectrograms/{file}").resize((128, 128)).convert('L')
        X.append(np.array(img) / 255.0)
        y.append(labels[file.split("_")[0]])
    return np.array(X)[..., np.newaxis], tf.keras.utils.to_categorical(y, num_classes=3)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("models/rf_classifier.h5")
