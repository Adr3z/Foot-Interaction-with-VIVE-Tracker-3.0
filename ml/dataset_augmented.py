"""
dataset_augmented.py
--------------------

Expands the base dataset via 4-way flip augmentation to increase training
sample diversity without collecting new recordings.

Each original pressure-map image is replicated into four variants:
    1. Original
    2. Horizontal flip (left ↔ right)
    3. Vertical flip (top ↔ bottom)
    4. Combined flip (horizontal + vertical)
"""


import os
import numpy as np

def create_augmented_dataset(
    input_path: str = "data/processed/dataset.npz",
    output_path: str = "data/processed/dataset_augmented.npz"
):
    if not os.path.exists(input_path):
        print(f"Error: Base dataset not found. Run dataset_builder first")
        return

    # 1. load all the recordings
    data = np.load(input_path)
    X_raw, y_raw = data["X"], data["y"]
    
    X_augmented = []
    y_augmented = []

    print(f"Processing {len(X_raw)} base recordings...")

    # 2. Apply 4-way flips to each recording
    for img, label in zip(X_raw, y_raw):
        # Flip 1: Original
        X_augmented.append(img)
        y_augmented.append(label)

        # Flip 2: Horizontal (left to right)
        X_augmented.append(np.flip(img, axis=1))
        y_augmented.append(label)

        # Flip 3: Vertical (top to bottom)
        X_augmented.append(np.flip(img, axis=0))
        y_augmented.append(label)

        # Flip 4: Combined (Horizontal + Vertical)
        X_augmented.append(np.flip(img, axis=(0, 1)))
        y_augmented.append(label)

    # 3. Convert to NumPy arrays
    X_augmented = np.array(X_augmented, dtype=np.float32)
    y_augmented = np.array(y_augmented, dtype=np.int32)

    # 4. Save the augmented dataset
    np.savez(output_path, X=X_augmented, y=y_augmented)

    print(f"\nAugmented dataset saved successfully to: {output_path}!")
    print(f"-> Total samples: {X_augmented.shape[0]}")
    print(f"-> Class distribution: {dict(zip(*np.unique(y_augmented, return_counts=True)))}")

if __name__ == "__main__":
    create_augmented_dataset()