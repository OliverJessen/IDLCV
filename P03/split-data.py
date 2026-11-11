
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# PH2 DATASET
# ============================================================

# Dataset directory
root_ph2 = Path("/dtu/datasets1/02516/PH2_Dataset_images")

# Get all valid image IDs
ph2_image_ids = []

for folder in sorted(root_ph2.iterdir()):
    if not folder.is_dir():
        continue

    image_id = folder.name

    image_path = (
        folder / f"{image_id}_Dermoscopic_Image"
        / f"{image_id}.bmp"
    )

    mask_path = (
        folder / f"{image_id}_lesion"
        / f"{image_id}_lesion.bmp"
    )

    if image_path.exists() and mask_path.exists():
        ph2_image_ids.append(image_id)

# First split: 70% training, 30% remaining
ph2_train_ids, ph2_remaining_ids = train_test_split(
    ph2_image_ids,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Second split: divide remaining 30% equally
ph2_val_ids, ph2_test_ids = train_test_split(
    ph2_remaining_ids,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# Save PH2 splits
ph2_splits = {
    "train": ph2_train_ids,
    "val": ph2_val_ids,
    "test": ph2_test_ids
}

with open("ph2_splits.json", "w") as f:
    json.dump(ph2_splits, f, indent=4)

# Verify no overlap
assert set(ph2_train_ids).isdisjoint(ph2_val_ids)
assert set(ph2_train_ids).isdisjoint(ph2_test_ids)
assert set(ph2_val_ids).isdisjoint(ph2_test_ids)

assert (
    len(ph2_train_ids) + len(ph2_val_ids) + len(ph2_test_ids)
    == len(ph2_image_ids)
)

print("PH2 DATASET")
print("Total images:", len(ph2_image_ids))
print("Training images:", len(ph2_train_ids))
print("Validation images:", len(ph2_val_ids))
print("Test images:", len(ph2_test_ids))
print("Splits saved to ph2_splits.json")


# ============================================================
# DRIVE DATASET
# ============================================================

# Dataset directory
root_drive = Path("/dtu/datasets1/02516/DRIVE")

# Get all images with vessel segmentation annotations
drive_image_ids = []

for image_path in sorted((root_drive / "training" / "images").glob("*.tif")):
    image_id = image_path.name.split("_")[0]

    mask_path = (
        root_drive / "training" / "1st_manual"
        / f"{image_id}_manual1.gif"
    )

    if mask_path.exists():
        drive_image_ids.append(image_id)

# First split: 70% training, 30% remaining
drive_train_ids, drive_remaining_ids = train_test_split(
    drive_image_ids,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Second split: divide remaining 30% equally
drive_val_ids, drive_test_ids = train_test_split(
    drive_remaining_ids,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# Save splits
drive_splits = {
    "train": drive_train_ids,
    "val": drive_val_ids,
    "test": drive_test_ids
}

with open("drive_splits.json", "w") as f:
    json.dump(drive_splits, f, indent=4)

# Verify no overlap
assert set(drive_train_ids).isdisjoint(drive_val_ids)
assert set(drive_train_ids).isdisjoint(drive_test_ids)
assert set(drive_val_ids).isdisjoint(drive_test_ids)

assert (
    len(drive_train_ids) + len(drive_val_ids) + len(drive_test_ids)
    == len(drive_image_ids)
)

print("\nDRIVE DATASET")
print("Total annotated images:", len(drive_image_ids))
print("Training images:", len(drive_train_ids))
print("Validation images:", len(drive_val_ids))
print("Test images:", len(drive_test_ids))
print("Splits saved to drive_splits.json")
