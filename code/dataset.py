"""Dataset for AgriVision DB-1 (Images/ + Masks/) and DB-3 (Generated_Images/ + Generated_Masks/)."""
import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def _collect(root, sub):
    files = []
    for e in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, sub, e)))
        files.extend(glob.glob(os.path.join(root, sub, "**", e), recursive=True))
    return sorted(set(files))


class AgriVisionSeg(Dataset):
    """Binary berry segmentation. Masks are binary (0/255)."""

    def __init__(self, root, image_sub="Images", mask_sub="Masks",
                 img_size=512, train=False, augment=False):
        self.root = Path(root)
        self.img_size = img_size
        self.train = train
        self.augment = augment
        self.images = _collect(self.root, image_sub)
        self.pairs = []
        for ip in self.images:
            stem = Path(ip).stem
            cand = list(Path(self.root, mask_sub).glob(stem + ".*"))
            if cand:
                self.pairs.append((ip, str(cand[0])))
        if not self.pairs:
            raise FileNotFoundError(f"No image/mask pairs under {root} ({image_sub}/{mask_sub})")

    def __len__(self):
        return len(self.pairs)

    def _load(self, ip, mp):
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        return img, mask

    def _resize(self, img, mask):
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST, antialias=False)
        return img, mask

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = TF.hflip(img); mask = TF.hflip(mask)
        if random.random() < 0.5:
            img = TF.vflip(img); mask = TF.vflip(mask)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        return img, mask

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img, mask = self._load(ip, mp)
        if self.augment:
            img, mask = self._augment(img, mask)
        img, mask = self._resize(img, mask)
        img = TF.to_tensor(img)
        mask = (TF.to_tensor(mask) > 0.5).float()
        return img, mask


def make_splits(ds, train_ratio=0.7, val_ratio=0.1, seed=42):
    n = len(ds)
    idx = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (torch.utils.data.Subset(ds, train_idx),
            torch.utils.data.Subset(ds, val_idx),
            torch.utils.data.Subset(ds, test_idx))
