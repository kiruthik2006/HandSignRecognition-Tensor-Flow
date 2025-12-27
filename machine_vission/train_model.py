import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load CSV with no header
df = pd.read_csv("gesture_landmarks.csv", header=None)

# Features (first 63 values) and labels (last value)
X = df.iloc[:, :-1].values.astype("float32")
y = df.iloc[:, -1].values

# Encode string labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save labels to file
with open("labels.txt", "w") as f:
    for label in label_encoder.classes_:
        f.write(f"{label}\n")

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Done! Saved gesture_model.tflite and labels.txt")
