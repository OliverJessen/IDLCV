import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.datasets import PH2Dataset

ROOT = "/dtu/datasets1/02516/PH2_Dataset_images"
STRATEGY = "center"         # "center" or "random"
IMAGE_ID = "IMD004"         # any image of the training split
N_SHOW = 10                 # clicks per class

with open(f"datasets/ph2_clicks_{STRATEGY}.json") as f:
    clicks = json.load(f)["clicks"][IMAGE_ID]

positive = np.array(clicks["positive"][:N_SHOW])
negative = np.array(clicks["negative"][:N_SHOW])

image, mask = PH2Dataset(ROOT, image_ids=[IMAGE_ID])[0]
image = image.permute(1, 2, 0).numpy()
mask = mask.squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(6, 3.3))

for ax, background, title in zip(axes, [image, mask], ["Image", "Mask"]):
    ax.imshow(background, cmap="gray")

    # scatter takes (x, y) = (col, row).
    # Different marker shapes, so the classes are not told apart by color only
    ax.scatter(positive[:, 1], positive[:, 0], marker="o", s=45,
               c="limegreen", edgecolors="black", linewidths=0.8,
               label="Positive click")
    ax.scatter(negative[:, 1], negative[:, 0], marker="X", s=55,
               c="red", edgecolors="white", linewidths=0.8,
               label="Negative click")

    ax.set_title(title, fontsize=10)
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           fontsize=9)

plt.tight_layout(rect=(0, 0.07, 1, 1))

path = f"results/ph2_click_example_{STRATEGY}.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()

print("Figure saved to", path)