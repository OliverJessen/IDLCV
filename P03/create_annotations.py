import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from datasets.datasets import PH2Dataset

ROOT = "/dtu/datasets1/02516/PH2_Dataset_images"
N_POS = 80          # positive clicks per image (on the lesion)
N_NEG = 80          # negative clicks per image (on the background)
N_EXAMPLES = 3      # number of example pictures
STRATEGY = "center"                         # "center" or "random"
POWER = 4 if STRATEGY == "center" else 0    # 0 = uniform random
SEED = 42

with open("datasets/ph2_splits.json") as f:
    splits = json.load(f)

# Weak annotations are only needed for the training set
dataset = PH2Dataset(ROOT, image_ids=splits["train"])
rng = np.random.default_rng(SEED)

clicks = {}

for idx, (image_path, _) in enumerate(dataset.samples):
    _, mask = dataset[idx]
    mask = mask.squeeze(0).numpy()

    points = {}

    for name, value, n in [("positive", 1, N_POS), ("negative", 0, N_NEG)]:
        # Pixels where the next click may be placed
        available = mask == value
        points[name] = []

        for _ in range(n):
            # Distance of every pixel to the nearest "forbidden" pixel:
            # the region border, the image border (through the padding)
            # or a click that was already placed.
            distance = distance_transform_edt(np.pad(available, 1))[1:-1, 1:-1]

            # A human clicks clearly inside the region and spreads the
            # clicks evenly, so pixels far away from the border AND from
            # earlier clicks are more likely. POWER controls how strong
            # this preference is (0 = uniform random).
            weights = (distance.ravel() > 0) * distance.ravel() ** POWER

            if weights.sum() == 0:      # no pixels left in this region
                break

            chosen = rng.choice(weights.size, p=weights / weights.sum())
            row, col = divmod(int(chosen), mask.shape[1])

            points[name].append([row, col])
            available[row, col] = False

    clicks[image_path.stem] = points

with open(f"datasets/ph2_clicks_{STRATEGY}.json", "w") as f:
    json.dump({"clicks": clicks}, f)

print("Clicks saved for", len(clicks), "training images")

# Example pictures: image with clicks (top) and mask with clicks (bottom)
fig, axes = plt.subplots(2, N_EXAMPLES, figsize=(4 * N_EXAMPLES, 8))

for col in range(N_EXAMPLES):
    image, mask = dataset[col]
    image_id = dataset.samples[col][0].stem

    positive = np.array(clicks[image_id]["positive"])
    negative = np.array(clicks[image_id]["negative"])

    backgrounds = [image.permute(1, 2, 0).numpy(), mask.squeeze(0).numpy()]

    for row, background in enumerate(backgrounds):
        ax = axes[row, col]
        ax.imshow(background, cmap="gray")

        # scatter takes (x, y) = (col, row)
        if len(positive):
            ax.scatter(positive[:, 1], positive[:, 0], c="lime",
                       edgecolors="black", s=60, label="positive")
        if len(negative):
            ax.scatter(negative[:, 1], negative[:, 0], c="red",
                       edgecolors="black", s=60, label="negative")

        ax.set_title(image_id)
        ax.axis("off")

axes[0, 0].legend(loc="upper left")
plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig(f"results/ph2_clicks_examples_{STRATEGY}.png", dpi=150)
plt.close()

print("Example pictures saved to results/ph2_clicks_examples.png")