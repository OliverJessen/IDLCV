
import json

from torch.utils.data import DataLoader
from datasets.datasets import PH2Dataset

root = "/dtu/datasets1/02516/PH2_Dataset_images"

# Load saved splits
with open("datasets/ph2_splits.json", "r") as f:
    splits = json.load(f)

# Create datasets
train_dataset = PH2Dataset(root, image_ids=splits["train"])
val_dataset = PH2Dataset(root, image_ids=splits["val"])
test_dataset = PH2Dataset(root, image_ids=splits["test"])

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# Test one training batch
images, masks = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch mask shape:", masks.shape)
