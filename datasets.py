from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class PH2Dataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256), image_ids=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.samples = []

        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue

            image_id = folder.name

            # Only load images belonging to the selected split
            if image_ids is not None and image_id not in image_ids:
                continue

            image_path = (
                folder / f"{image_id}_Dermoscopic_Image"
                / f"{image_id}.bmp"
            )

            mask_path = (
                folder / f"{image_id}_lesion"
                / f"{image_id}_lesion.bmp"
            )

            if image_path.exists() and mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError("No valid PH2 images found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST
        )

        image = TF.to_tensor(image)

        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class DRIVEDataset(Dataset):
    def __init__(self, root_dir, split="training",
                 image_size=(512, 512), image_ids=None):

        self.root_dir = Path(root_dir) / split
        self.image_size = image_size
        self.samples = []

        image_folder = self.root_dir / "images"
        mask_folder = self.root_dir / "1st_manual"

        for image_path in sorted(image_folder.glob("*.tif")):
            image_id = image_path.name.split("_")[0]

            if image_ids is not None and image_id not in image_ids:
                continue

            mask_path = mask_folder / f"{image_id}_manual1.gif"

            if image_path.exists() and mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError("No valid DRIVE images found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST
        )

        image = TF.to_tensor(image)

        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
