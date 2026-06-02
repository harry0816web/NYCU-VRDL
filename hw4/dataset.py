"""Datasets for HW4 image restoration (rain + snow)."""

import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


# --------------------------------------------------------------------------- #
#  Data augmentation
# --------------------------------------------------------------------------- #

def _augment(img1, img2, mode):
    """Apply the same geometric augmentation to two HWC numpy arrays."""
    if mode == 0:
        return img1, img2
    elif mode == 1:  # horizontal flip
        return np.fliplr(img1).copy(), np.fliplr(img2).copy()
    elif mode == 2:  # vertical flip
        return np.flipud(img1).copy(), np.flipud(img2).copy()
    elif mode == 3:  # rotate 90
        return np.rot90(img1, 1).copy(), np.rot90(img2, 1).copy()
    elif mode == 4:  # rotate 90 + h-flip
        r1, r2 = np.rot90(img1, 1), np.rot90(img2, 1)
        return np.fliplr(r1).copy(), np.fliplr(r2).copy()
    elif mode == 5:  # rotate 180
        return np.rot90(img1, 2).copy(), np.rot90(img2, 2).copy()
    elif mode == 6:  # rotate 270
        return np.rot90(img1, 3).copy(), np.rot90(img2, 3).copy()
    elif mode == 7:  # rotate 270 + h-flip
        r1, r2 = np.rot90(img1, 3), np.rot90(img2, 3)
        return np.fliplr(r1).copy(), np.fliplr(r2).copy()
    return img1, img2


def _channel_shuffle(img1, img2):
    """A1: Randomly permute RGB channels (same permutation for both images)."""
    perm = np.random.permutation(3)
    return img1[:, :, perm].copy(), img2[:, :, perm].copy()


def _random_grayscale(img1, img2):
    """A3: Convert both images to grayscale (replicated to 3 channels)."""
    # ITU-R BT.601 luma weights
    w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    g1 = np.dot(img1.astype(np.float32), w)
    g2 = np.dot(img2.astype(np.float32), w)
    g1 = np.clip(g1, 0, 255).astype(np.uint8)
    g2 = np.clip(g2, 0, 255).astype(np.uint8)
    return np.stack([g1, g1, g1], axis=-1), np.stack([g2, g2, g2], axis=-1)


# --------------------------------------------------------------------------- #
#  Training Dataset
# --------------------------------------------------------------------------- #

class TrainDataset(Dataset):
    """Paired degraded-clean dataset for rain + snow restoration.

    Scans ``degraded_dir`` for files matching ``rain-*.png`` and
    ``snow-*.png``, then finds the corresponding clean image in
    ``clean_dir`` (e.g. ``rain_clean-*.png``).

    Args:
        degraded_dir: Path to ``data/train/degraded``.
        clean_dir: Path to ``data/train/clean``.
        patch_size: Random crop size (default 128). Set 0 or >= image size
                    to disable cropping (used for validation).
        augment: Whether to apply random flip / rotation.
        cache: If True, pre-load all images into RAM (~600 MB for 3200 pairs).
    """

    def __init__(self, degraded_dir, clean_dir, patch_size=128, augment=True,
                 cache=False, channel_shuffle=False, cutmix=False,
                 cutmix_alpha=1.0, random_grayscale=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.augment = augment
        self.cache = cache
        self.channel_shuffle = channel_shuffle
        self.cutmix = cutmix
        self.cutmix_alpha = cutmix_alpha
        self.random_grayscale = random_grayscale
        self.to_tensor = ToTensor()

        self.pairs = []  # list of (degraded_path, clean_path, deg_type)

        for fname in sorted(os.listdir(degraded_dir)):
            if not fname.endswith(".png"):
                continue

            deg_path = os.path.join(degraded_dir, fname)

            # rain-123.png -> rain_clean-123.png
            if fname.startswith("rain-"):
                idx = fname.replace("rain-", "").replace(".png", "")
                clean_name = f"rain_clean-{idx}.png"
                deg_type = "rain"
            elif fname.startswith("snow-"):
                idx = fname.replace("snow-", "").replace(".png", "")
                clean_name = f"snow_clean-{idx}.png"
                deg_type = "snow"
            else:
                continue

            clean_path = os.path.join(clean_dir, clean_name)
            if os.path.exists(clean_path):
                self.pairs.append((deg_path, clean_path, deg_type))

        # In-memory cache
        self._cache_data = None
        if cache:
            print(f"[TrainDataset] Caching {len(self.pairs)} pairs into RAM ...")
            self._cache_data = []
            for deg_path, clean_path, deg_type in self.pairs:
                deg_img = np.array(Image.open(deg_path).convert("RGB"))
                cln_img = np.array(Image.open(clean_path).convert("RGB"))
                self._cache_data.append((deg_img, cln_img))
            print("[TrainDataset] Cache complete.")

        n_rain = sum(1 for _, _, t in self.pairs if t == "rain")
        n_snow = sum(1 for _, _, t in self.pairs if t == "snow")
        print(f"[TrainDataset] Loaded {len(self.pairs)} pairs "
              f"(rain={n_rain}, snow={n_snow}), "
              f"patch={patch_size}, augment={augment}, cache={cache}")

    def __len__(self):
        return len(self.pairs)

    def _load_pair(self, idx):
        """Load a single degraded/clean pair by index."""
        if self._cache_data is not None:
            return self._cache_data[idx][0].copy(), self._cache_data[idx][1].copy()
        deg_path, clean_path, _ = self.pairs[idx]
        degrad_img = np.array(Image.open(deg_path).convert("RGB"))
        clean_img = np.array(Image.open(clean_path).convert("RGB"))
        return degrad_img, clean_img

    def __getitem__(self, idx):
        deg_path, clean_path, deg_type = self.pairs[idx]

        degrad_img, clean_img = self._load_pair(idx)

        H, W, _ = degrad_img.shape

        # Random crop
        if self.patch_size > 0 and (H > self.patch_size or W > self.patch_size):
            rh = random.randint(0, H - self.patch_size)
            rw = random.randint(0, W - self.patch_size)
            degrad_img = degrad_img[rh:rh + self.patch_size,
                                    rw:rw + self.patch_size]
            clean_img = clean_img[rh:rh + self.patch_size,
                                  rw:rw + self.patch_size]

        # Augmentation
        if self.augment:
            mode = random.randint(0, 7)
            degrad_img, clean_img = _augment(degrad_img, clean_img, mode)

        # A1: RGB Channel Shuffle
        if self.channel_shuffle and random.random() < 0.5:
            degrad_img, clean_img = _channel_shuffle(degrad_img, clean_img)

        # A3: Random Grayscale
        if self.random_grayscale > 0 and random.random() < self.random_grayscale:
            degrad_img, clean_img = _random_grayscale(degrad_img, clean_img)

        # A2: CutMix — mix a rectangular region from another same-type sample
        if self.cutmix and random.random() < 0.5:
            degrad_img, clean_img = self._apply_cutmix(
                degrad_img, clean_img, deg_type,
            )

        # To tensor: HWC uint8 -> CHW float [0, 1]
        degrad_tensor = self.to_tensor(degrad_img)
        clean_tensor = self.to_tensor(clean_img)

        return degrad_tensor, clean_tensor, deg_type

    # --- A2: CutMix helper ------------------------------------------------ #

    # Build per-type index lists lazily (built once, on first call)
    _type_indices = None

    def _get_type_indices(self):
        if self._type_indices is None:
            self._type_indices = {}
            for i, (_, _, t) in enumerate(self.pairs):
                self._type_indices.setdefault(t, []).append(i)
        return self._type_indices

    def _apply_cutmix(self, degrad_img, clean_img, deg_type):
        """Mix a random rectangular region from another same-type sample."""
        type_idx = self._get_type_indices()
        candidates = type_idx.get(deg_type, [])
        if len(candidates) < 2:
            return degrad_img, clean_img

        other_idx = random.choice(candidates)
        other_deg, other_cln = self._load_pair(other_idx)

        h, w, _ = degrad_img.shape
        oh, ow, _ = other_deg.shape

        # Crop other image to same patch_size if needed
        if oh >= h and ow >= w:
            rh2 = random.randint(0, oh - h)
            rw2 = random.randint(0, ow - w)
            other_deg = other_deg[rh2:rh2 + h, rw2:rw2 + w]
            other_cln = other_cln[rh2:rh2 + h, rw2:rw2 + w]
        else:
            return degrad_img, clean_img  # skip if other image is smaller

        # Sample CutMix box
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        if cut_h < 1 or cut_w < 1:
            return degrad_img, clean_img

        cy = random.randint(0, h - cut_h)
        cx = random.randint(0, w - cut_w)

        degrad_img = degrad_img.copy()
        clean_img = clean_img.copy()
        degrad_img[cy:cy + cut_h, cx:cx + cut_w] = other_deg[cy:cy + cut_h, cx:cx + cut_w]
        clean_img[cy:cy + cut_h, cx:cx + cut_w] = other_cln[cy:cy + cut_h, cx:cx + cut_w]

        return degrad_img, clean_img


# --------------------------------------------------------------------------- #
#  Validation Dataset
# --------------------------------------------------------------------------- #

class ValDataset(Dataset):
    """Validation dataset — full-resolution, no crop, no augmentation.

    Uses the same file pairing logic as TrainDataset, but always returns
    the complete 256x256 image without any random transformation. This
    ensures validation PSNR reflects real test-time performance.

    Args:
        pairs: List of (degraded_path, clean_path, deg_type) tuples.
               Typically obtained from ``train_val_split``.
        cache_from: Optional list of (deg_np, cln_np) arrays from
                    TrainDataset's cache to avoid re-reading.
    """

    def __init__(self, pairs, cache_from=None):
        super().__init__()
        self.pairs = pairs
        self.to_tensor = ToTensor()

        # Pre-load for validation (small set, ~160 images)
        self._data = []
        for i, (deg_path, clean_path, deg_type) in enumerate(pairs):
            if cache_from is not None:
                deg_img, cln_img = cache_from[i]
            else:
                deg_img = np.array(Image.open(deg_path).convert("RGB"))
                cln_img = np.array(Image.open(clean_path).convert("RGB"))
            self._data.append((deg_img, cln_img, deg_type))

        n_rain = sum(1 for _, _, t in pairs if t == "rain")
        n_snow = sum(1 for _, _, t in pairs if t == "snow")
        print(f"[ValDataset] {len(pairs)} pairs (rain={n_rain}, snow={n_snow}), "
              f"no crop, no augment")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        deg_img, cln_img, deg_type = self._data[idx]
        return self.to_tensor(deg_img), self.to_tensor(cln_img), deg_type


# --------------------------------------------------------------------------- #
#  Test Dataset
# --------------------------------------------------------------------------- #

class TestDataset(Dataset):
    """Test dataset — loads degraded images without clean targets.

    Scans ``degraded_dir`` for ``*.png`` files (0.png ~ 99.png).
    Returns the image tensor and filename for building pred.npz.
    """

    def __init__(self, degraded_dir):
        super().__init__()
        self.to_tensor = ToTensor()

        self.images = []  # list of (path, filename)
        for fname in sorted(os.listdir(degraded_dir)):
            if fname.endswith(".png"):
                self.images.append(
                    (os.path.join(degraded_dir, fname), fname)
                )

        print(f"[TestDataset] Loaded {len(self.images)} test images "
              f"from {degraded_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, fname = self.images[idx]
        img = np.array(Image.open(path).convert("RGB"))
        tensor = self.to_tensor(img)
        return tensor, fname


# --------------------------------------------------------------------------- #
#  Utility: stratified train / val split
# --------------------------------------------------------------------------- #

def train_val_split(dataset, val_ratio=0.05, seed=42):
    """Stratified split of a TrainDataset into train and val subsets.

    Splits rain and snow indices separately so that the val set has the
    same rain/snow ratio as the full dataset. Returns:
      - train_indices: list of int
      - val_pairs: list of (deg_path, clean_path, deg_type)
      - val_cache: list of (deg_np, cln_np) or None

    The caller should create a ``Subset(dataset, train_indices)`` for
    training and a ``ValDataset(val_pairs, val_cache)`` for validation.
    """
    rng = random.Random(seed)

    # Separate indices by degradation type
    rain_indices = [i for i, (_, _, t) in enumerate(dataset.pairs) if t == "rain"]
    snow_indices = [i for i, (_, _, t) in enumerate(dataset.pairs) if t == "snow"]

    rng.shuffle(rain_indices)
    rng.shuffle(snow_indices)

    # Stratified split
    n_rain_val = max(1, int(len(rain_indices) * val_ratio))
    n_snow_val = max(1, int(len(snow_indices) * val_ratio))

    val_indices = rain_indices[:n_rain_val] + snow_indices[:n_snow_val]
    train_indices = rain_indices[n_rain_val:] + snow_indices[n_snow_val:]

    # Build val pairs (and cache if available)
    val_pairs = [dataset.pairs[i] for i in val_indices]
    val_cache = None
    if dataset._cache_data is not None:
        val_cache = [dataset._cache_data[i] for i in val_indices]

    return train_indices, val_pairs, val_cache
