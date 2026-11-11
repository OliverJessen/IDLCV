
from datasets import DRIVEDataset

root = "/dtu/datasets1/02516/DRIVE"

train_dataset = DRIVEDataset(root, split="training")

print("Number of training images:", len(train_dataset))

image, mask = train_dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask values:", mask.unique())
