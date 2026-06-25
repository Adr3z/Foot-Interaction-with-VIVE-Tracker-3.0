import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/processed/dataset.npz")
X, y = data["X"], data["y"]

fig, axes = plt.subplots(3, 6, figsize=(14, 7))
for cls_idx, cls in enumerate([1, 2, 3]):
    samples = X[y == cls][:6]
    for i, ax in enumerate(axes[cls_idx]):
        if i < len(samples):
            ax.imshow(samples[i], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Class {cls} [{i}]", fontsize=8)
        ax.axis("off")

plt.tight_layout()
plt.savefig("data/processed/dataset_preview.png", dpi=120)
plt.show()
print("saved to data/processed/dataset_preview.png")