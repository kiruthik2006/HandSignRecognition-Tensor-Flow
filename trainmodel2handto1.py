import pandas as pd
import string

# Load CSV
df = pd.read_csv("Indian Sign Language Gesture Landmarks.csv")

# Filter only right-hand x and y columns (ignore z)
right_xy_cols = [col for col in df.columns if "right_hand_" in col and not "_z_" in col]
clean_df = df[["target"] + right_xy_cols]

# Rename columns to x0, y0, ..., x20, y20
new_cols = []
for i in range(21):
    new_cols.append(f'x{i}')
    new_cols.append(f'y{i}')
clean_df.columns = ['label'] + new_cols

# Map numeric labels 0–25 to A–Z
label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase)}
clean_df['label'] = clean_df['label'].map(label_map)

# Save cleaned CSV
clean_df.to_csv("clean_isl_landmarks.csv", index=False)
print("✅ Cleaned CSV saved as 'clean_isl_landmarks.csv'")
