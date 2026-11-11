
from datasets import PH2Dataset

root = "/dtu/datasets1/02516/PH2_Dataset_images"

dataset = PH2Dataset(root)

print("Number of images:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask values:", mask.unique())
