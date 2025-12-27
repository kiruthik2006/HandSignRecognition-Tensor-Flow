import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load CSV without headers
df = pd.read_csv("HandLandmarks_Labeled_Corrected.csv", header=None)

# Assign columns
df.columns = ["label", "handedness"] + [f"lm_{i}" for i in range(63)]

# Features and labels
X = df.iloc[:, 2:].values
y = df["label"].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, validation_split=0.1)

# Save TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

# Save label map
with open("labels.txt", "w") as f:
    for label in encoder.classes_:
        f.write(label + "\n")

print("✅ Training complete. Model and labels saved.")
