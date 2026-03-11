"""
dataset.py — Data loading and preprocessing for PatchCL-AE
==========================================================

Handles:
  1. Loading the Brain Tumor MRI Dataset (Kaggle 4-class: glioma,
     meningioma, pituitary, notumor).
  2. For TRAINING: only ``notumor`` images are used (the model never sees
     anomalous data during training).
  3. For TESTING: all four classes are loaded; ``notumor`` → label 0 (normal),
     everything else → label 1 (anomaly).
  4. Images are resized to 256×256, normalised to [-1, 1] (matching the
     Tanh decoder output range).
  5. A denoising auto-encoder noise schedule adds Gaussian noise clamped to
     [-1, 1] for the training split.

Expected folder layout (Kaggle default names):
    data/
      Training/
        notumor/
        glioma/
        meningioma/
        pituitary/
      Testing/
        notumor/
        glioma/
        meningioma/
        pituitary/
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


NORMAL_FOLDER = "notumor"   # case-insensitive match below


def _is_image(fname: str) -> bool:
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for the Brain Tumor MRI dataset.

    For training (split='Training'):
        - Loads ONLY ``notumor`` images (normal class).
        - Returns ``(noisy_image, clean_image)`` pairs for the denoising AE.
    For testing (split='Testing'):
        - Loads ALL classes.  ``notumor`` → label 0, others → label 1.
        - Returns ``(image, label)`` where label ∈ {0, 1}.
    """

    def __init__(self, root: str, split: str = "Training",
                 image_size: int = 256, noise_std: float = 0.05):
        super().__init__()
        self.split = split
        self.noise_std = noise_std

        self.image_paths: list[str] = []
        self.labels: list[int] = []

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for cls_name in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            is_normal = cls_name.lower() == NORMAL_FOLDER.lower()
            label = 0 if is_normal else 1

            # Training: skip anomaly folders entirely
            if split == "Training" and not is_normal:
                continue

            for fname in sorted(os.listdir(cls_dir)):
                if _is_image(fname):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

        # Resize + ToTensor → [0,1] then Normalize → [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        clean = self.transform(img)  # (3, H, W) in [-1, 1]

        if self.split == "Training":
            noise = torch.randn_like(clean) * self.noise_std
            noisy = torch.clamp(clean + noise, -1.0, 1.0)
            return noisy, clean
        else:
            return clean, self.labels[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(data_root: str = "./data", batch_size: int = 8,
                    num_workers: int = 2, noise_std: float = 0.05,
                    image_size: int = 256):
    """
    Build train and test DataLoaders.

    Expects ``data_root`` to contain ``Training/`` and ``Testing/`` folders
    with the standard Kaggle Brain Tumor MRI structure.
    """
    train_ds = BrainTumorDataset(data_root, split="Training",
                                 image_size=image_size, noise_std=noise_std)
    test_ds = BrainTumorDataset(data_root, split="Testing",
                                image_size=image_size, noise_std=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
