import numpy as np

labels = np.load("labels.npy", allow_pickle=True)
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(str(label) + "\n")
