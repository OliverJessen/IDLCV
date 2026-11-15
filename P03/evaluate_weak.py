import json

import torch
from torch.utils.data import DataLoader

from datasets.datasets import PH2Dataset
from models import UNetImproved     # use the same import line as in train_weak.py

# Configuration
CHECKPOINT = "checkpoints/PH2_unet_improved_best.pth"

ROOT = "/dtu/datasets1/02516/PH2_Dataset_images"
SPLIT_FILE = "datasets/ph2_splits.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetImproved().to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()


def compute_metrics(logits, targets, eps=1e-6):
    """Five segmentation metrics, computed per image. Returns shape (batch, 5)."""
    predictions = (torch.sigmoid(logits) > 0.5).float()

    tp = (predictions * targets).sum(dim=(1, 2, 3))
    fp = (predictions * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - predictions) * targets).sum(dim=(1, 2, 3))
    tn = ((1 - predictions) * (1 - targets)).sum(dim=(1, 2, 3))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return torch.stack([dice, iou, accuracy, sensitivity, specificity], dim=1)


with open(SPLIT_FILE) as f:
    splits = json.load(f)

print("Checkpoint:", CHECKPOINT)
print(f"{'Split':<6} {'Dice':>8} {'IoU':>8} {'Acc':>8} {'Sens':>8} {'Spec':>8}")

for split in ["train", "val", "test"]:
    # Always evaluated against the full masks
    dataset = PH2Dataset(ROOT, image_ids=splits[split])
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    results = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            results.append(compute_metrics(model(images), masks))

    # Average over all images of the split
    mean = torch.cat(results).mean(dim=0)

    print(f"{split:<6} " + " ".join(f"{value:8.4f}" for value in mean))