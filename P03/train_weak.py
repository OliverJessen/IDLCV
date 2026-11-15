
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.datasets import PH2Dataset, DRIVEDataset
from models import EncoderDecoderCNN, EncoderDecoderImproved, UNet, UNetImproved

# Configuration
DATASET = "PH2"          # "PH2" or "DRIVE"
MODEL = "unet_improved"           # "unet" or "encoder_decoder"

CLICKS_FILE = "datasets/ph2_clicks_center.json"
N_POS = 10                  # positive clicks used per image
N_NEG = 10                  # negative clicks used per image

EPOCHS = 50
LEARNING_RATE = 1e-4
SEED = 42

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Dataset with point-level labels instead of full masks
class PH2ClickDataset(Dataset):
    def __init__(self, dataset, clicks_file, n_pos, n_neg):
        self.dataset = dataset
        self.n_pos = n_pos
        self.n_neg = n_neg

        with open(clicks_file) as f:
            self.clicks = json.load(f)["clicks"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image_id = self.dataset.samples[idx][0].stem
        points = self.clicks[image_id]

        # -1 = no annotation, 1 = positive click, 0 = negative click
        labels = torch.full_like(mask, -1.0)

        for row, col in points["positive"][:self.n_pos]:
            labels[0, row, col] = 1.0
        for row, col in points["negative"][:self.n_neg]:
            labels[0, row, col] = 0.0

        return image, labels

# Load dataset splits
if DATASET == "PH2":
    root = "/dtu/datasets1/02516/PH2_Dataset_images"
    split_file = "datasets/ph2_splits.json"
    batch_size = 4

elif DATASET == "DRIVE":
    root = "/dtu/datasets1/02516/DRIVE"
    split_file = "datasets/drive_splits.json"
    batch_size = 2

else:
    raise ValueError("Unknown dataset")

with open(split_file) as f:
    splits = json.load(f)

# Create datasets
if DATASET == "PH2":
    train_dataset = PH2ClickDataset(
        PH2Dataset(root, image_ids=splits["train"]),
        CLICKS_FILE, N_POS, N_NEG
    )
    val_dataset = PH2Dataset(root, image_ids=splits["val"])

else:
    train_dataset = DRIVEDataset(
        root, split="training", image_ids=splits["train"]
    )
    val_dataset = DRIVEDataset(
        root, split="training", image_ids=splits["val"]
    )

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

# Create model
if MODEL == "unet":
    model = UNet().to(device)
elif MODEL == "unet_improved":
    model = UNetImproved().to(device)
elif MODEL == "encoder_decoder":
    model = EncoderDecoderCNN().to(device)
elif MODEL == "encoder_decoder_improved":
    model = EncoderDecoderImproved().to(device)
else:
    raise ValueError("Unknown model")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss for point-level supervision
def point_bce_loss(logits, labels):
    valid = labels >= 0          # True only at the clicked pixels
    return F.binary_cross_entropy_with_logits(logits[valid], labels[valid])

# Dice score for validation
def dice_score(logits, targets, smooth=1e-6):
    predictions = (torch.sigmoid(logits) > 0.5).float()

    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    total = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    return ((2 * intersection + smooth) / (total + smooth)).mean()


# Training
best_dice = -1.0
Path("checkpoints").mkdir(exist_ok=True)

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = point_bce_loss(outputs, masks) # new for annotations model

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_dice += dice_score(outputs, masks).item() * images.size(0)

    val_loss /= len(val_dataset)
    val_dice /= len(val_dataset)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}",
        flush=True
    )

    # Save best model
    if val_dice > best_dice:
        best_dice = val_dice

        torch.save(
            model.state_dict(),
            f"checkpoints/{DATASET}_{MODEL}_clicks_{N_POS}pos_{N_NEG}neg_best.pth"
        )

print(f"\nBest validation Dice: {best_dice:.4f}")
