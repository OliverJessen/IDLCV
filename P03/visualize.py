

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datasets import PH2Dataset, DRIVEDataset

#root = "/dtu/datasets1/02516/DRIVE"
root = "/dtu/datasets1/02516/PH2_Dataset_images"
#dataset = DRIVEDataset(root)
dataset = PH2Dataset(root)

image, mask = dataset[0]

image = image.permute(1, 2, 0).numpy()
mask = mask.squeeze(0).numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[0].set_title("Original image")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Ground truth mask")

axes[2].imshow(image)
axes[2].imshow(mask, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
axes[2].set_title("Image + mask overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("results/ph2_visualization.png", dpi=150)
plt.close()

print("Visualization saved!")
