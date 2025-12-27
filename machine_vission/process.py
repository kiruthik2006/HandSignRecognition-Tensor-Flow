import os
import csv
import cv2
import mediapipe as mp
import concurrent.futures

DATASET_DIR = "gestures"  # structure: gestures/palm/, gestures/fist/, etc.
CSV_OUT = "gesture_landmarks.csv"

def process_image(img_path, gesture):
    with mp.solutions.hands.Hands(static_image_mode=True) as mp_hands:
        image = cv2.imread(img_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(gesture)
            return row
    return None

# Prepare output
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for gesture in os.listdir(DATASET_DIR):
            gesture_path = os.path.join(DATASET_DIR, gesture)
            if not os.path.isdir(gesture_path) or gesture.startswith('.'):
                continue
            for img_file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_file)
                futures.append(executor.submit(process_image, img_path, gesture))

        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            if row:
                writer.writerow(row)
