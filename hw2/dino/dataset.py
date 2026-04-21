"""
COCO-format dataset and transforms for DETR.
"""
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

from util.box_ops import box_xyxy_to_cxcywh


# ---------------------------------------------------------------------------
# Transforms (image + bbox aware)
# ---------------------------------------------------------------------------

def resize(image, target, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(
                        max_size *
                        min_original_size /
                        max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            # input tuple is (w, h) to match PIL's size convention
            return size[1], size[0]
        return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig)
                   for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * \
            torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


class ColorJitter(object):
    def __init__(
            self,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=1.0):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target
        return self.jitter(img), target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomResizeScale(object):
    """Resize by a random scale factor sampled from [min_scale, max_scale]."""

    def __init__(self, min_scale=0.8, max_scale=1.0):
        assert 0 < min_scale <= max_scale
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, target=None):
        scale = random.uniform(self.min_scale, self.max_scale)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        return resize(img, target, (new_w, new_h), max_size=None)


def _filter_boxes(target, img_w, img_h, min_visibility=0.5):
    """Clip boxes and discard low-visibility ones.

    Applies the same keep mask to boxes, labels, area, and iscrowd.
    """
    if target is None or "boxes" not in target or len(target["boxes"]) == 0:
        return target
    target = target.copy()
    boxes = target["boxes"]
    orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    boxes[:, 0::2].clamp_(min=0, max=img_w)
    boxes[:, 1::2].clamp_(min=0, max=img_h)
    clipped_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    valid_box = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    visible_enough = (
        clipped_areas /
        orig_areas.clamp(
            min=1e-6)) >= min_visibility
    keep = valid_box & visible_enough

    target["boxes"] = boxes[keep]
    target["labels"] = target["labels"][keep]
    if "area" in target:
        target["area"] = target["area"][keep]
    if "iscrowd" in target:
        target["iscrowd"] = target["iscrowd"][keep]
    return target


class RandomRotation(object):
    """Bbox-aware rotation + translation with compensatory scale-up."""

    def __init__(
            self,
            degrees=10.0,
            translate=(
                0.1,
                0.1),
            min_visibility=0.5,
            p=1.0):
        self.degrees = degrees
        self.translate = translate
        self.min_visibility = min_visibility
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target
        w, h = img.size
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate[0], self.translate[0]) * w
        ty = random.uniform(-self.translate[1], self.translate[1]) * h

        theta = math.radians(abs(angle))
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        scale = cos_a + sin_a * max(h / w, w / h)

        img = F.affine(img, angle=angle, translate=[int(tx), int(ty)],
                       scale=scale, shear=0, fill=0)

        if target is not None and "boxes" in target and len(
                target["boxes"]) > 0:
            target = target.copy()
            boxes = target["boxes"].clone().float()
            cx, cy = w / 2.0, h / 2.0

            x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            corners_x = torch.stack([x0, x1, x1, x0], dim=1)  # (N, 4)
            corners_y = torch.stack([y0, y0, y1, y1], dim=1)  # (N, 4)

            corners_x -= cx
            corners_y -= cy

            rad = math.radians(-angle)  # F.affine uses counter-clockwise
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            new_x = scale * (cos_r * corners_x - sin_r * corners_y) + tx + cx
            new_y = scale * (sin_r * corners_x + cos_r * corners_y) + ty + cy

            target["boxes"] = torch.stack([
                new_x.min(dim=1).values,
                new_y.min(dim=1).values,
                new_x.max(dim=1).values,
                new_y.max(dim=1).values,
            ], dim=1)
            target = _filter_boxes(target, w, h, self.min_visibility)

        return img, target


class RandomPerspective(object):
    """Bbox-aware perspective transform via 4-corner perturbation."""

    def __init__(self, distortion_scale=0.1, p=0.5, min_visibility=0.5):
        self.distortion_scale = distortion_scale
        self.p = p
        self.min_visibility = min_visibility

    @staticmethod
    def _compute_homography(src, dst):
        """Compute 3x3 homography H such that dst = H @ src (homogeneous coords).
        src, dst: (4, 2) numpy arrays of corresponding points."""
        A = []
        for i in range(4):
            sx, sy = src[i]
            dx, dy = dst[i]
            A.append([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
            A.append([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])
        A = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]
        return H

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        w, h = img.size
        half_d = self.distortion_scale * min(w, h)

        tl = [random.uniform(-half_d, half_d), random.uniform(-half_d, half_d)]
        tr = [w - 1 + random.uniform(-half_d, half_d),
              random.uniform(-half_d, half_d)]
        br = [w - 1 + random.uniform(-half_d, half_d),
              h - 1 + random.uniform(-half_d, half_d)]
        bl = [random.uniform(-half_d, half_d), h - 1 +
              random.uniform(-half_d, half_d)]

        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [tl, tr, br, bl]

        img = F.perspective(img, startpoints, endpoints, fill=0)

        if target is not None and "boxes" in target and len(
                target["boxes"]) > 0:
            target = target.copy()

            src = np.array(startpoints, dtype=np.float64)
            dst = np.array(endpoints, dtype=np.float64)
            H = self._compute_homography(src, dst)
            H_t = torch.from_numpy(H).float()

            boxes = target["boxes"].clone().float()
            x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            corners_x = torch.stack([x0, x1, x1, x0], dim=1)  # (N, 4)
            corners_y = torch.stack([y0, y0, y1, y1], dim=1)  # (N, 4)

            N = boxes.shape[0]
            ones = torch.ones(N, 4)
            pts = torch.stack([corners_x, corners_y, ones], dim=2)  # (N, 4, 3)

            transformed = torch.einsum('ij,nkj->nki', H_t, pts)  # (N, 4, 3)
            transformed_x = transformed[:, :, 0] / transformed[:, :, 2]
            transformed_y = transformed[:, :, 1] / transformed[:, :, 2]

            target["boxes"] = torch.stack([
                transformed_x.min(dim=1).values,
                transformed_y.min(dim=1).values,
                transformed_x.max(dim=1).values,
                transformed_y.max(dim=1).values,
            ], dim=1)
            target = _filter_boxes(target, w, h, self.min_visibility)

        return img, target


class RandomCrop(object):
    """Random crop preserving aspect ratio and minimum box visibility."""

    def __init__(self, scale=(0.6, 1.0), p=0.5, min_visibility=0.5):
        self.scale = scale
        self.p = p
        self.min_visibility = min_visibility

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        w, h = img.size
        s = random.uniform(self.scale[0], self.scale[1])
        cw = max(1, int(w * s))
        ch = max(1, int(h * s))
        x0 = random.randint(0, w - cw) if w > cw else 0
        y0 = random.randint(0, h - ch) if h > ch else 0
        img = F.crop(img, y0, x0, ch, cw)

        if target is not None and "boxes" in target and len(
                target["boxes"]) > 0:
            target = target.copy()
            boxes = target["boxes"].clone().float()
            boxes[:, 0::2] -= x0
            boxes[:, 1::2] -= y0
            target["boxes"] = boxes
            target = _filter_boxes(target, cw, ch, self.min_visibility)
            target["size"] = torch.tensor([ch, cw])
        return img, target


class ISONoise(object):
    """Albumentations-style ISO noise: hue shift + multiplicative noise in HSV."""

    def __init__(self, p=0.2, intensity=0.05, color_shift=0.05):
        self.p = p
        self.intensity = intensity
        self.color_shift = color_shift

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target
        arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        arr[..., 0] = (arr[..., 0]
                       + np.random.uniform(-self.color_shift, self.color_shift)) % 1.0
        noise = 1.0 + np.random.randn(*arr.shape[:2]) * self.intensity
        arr[..., 1] = np.clip(arr[..., 1] * noise, 0, 1)
        arr[..., 2] = np.clip(arr[..., 2] * noise, 0, 1)
        arr = (arr * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="HSV").convert("RGB")
        return img, target


class GaussianBlur(object):
    """Random Gaussian blur (bbox-transparent)."""

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# ---------------------------------------------------------------------------
# COCO annotation converter
# ---------------------------------------------------------------------------

class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # COCO bbox is [x, y, w, h] -> convert to [x0, y0, x1, y1]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def make_coco_transforms(image_set, config=None):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if image_set == "train":
        short_edge = config.get('train_short_edge', 320) if config else 320
        max_size = config.get('train_max_size', 640) if config else 640
        scale_min = config.get(
            'train_random_scale_min',
            0.9) if config else 0.9
        scale_max = config.get(
            'train_random_scale_max',
            1.0) if config else 1.0

        min_vis = config.get('augment_min_visibility', 0.5) if config else 0.5
        degrees = config.get('augment_degrees', 10.0) if config else 10.0
        translate = config.get(
            'augment_translate', [
                0.1, 0.1]) if config else [
            0.1, 0.1]
        persp_scale = config.get(
            'augment_perspective_scale',
            0.1) if config else 0.1
        persp_p = config.get('augment_perspective_p', 0.5) if config else 0.5
        blur_kernel = config.get('augment_blur_kernel', 5) if config else 5
        blur_sigma = config.get(
            'augment_blur_sigma', [
                0.1, 2.0]) if config else [
            0.1, 2.0]
        blur_p = config.get('augment_blur_p', 0.5) if config else 0.5

        iso_enabled = config.get('aug_iso_noise', False) if config else False
        iso_p = config.get('aug_iso_noise_p', 0.2) if config else 0.2
        iso_intensity = config.get(
            'aug_iso_noise_intensity',
            0.05) if config else 0.05
        crop_p = config.get('aug_crop_p', 0.5) if config else 0.5
        crop_scale = config.get(
            'aug_crop_scale', [
                0.6, 1.0]) if config else [
            0.6, 1.0]
        color_jitter_p = config.get(
            'aug_color_jitter_p',
            1.0) if config else 1.0
        rot_p = config.get('aug_rotation_p', 1.0) if config else 1.0

        transforms_list = [
            # Phase 1: Pixel-level transformations (lens-blur first, then
            # sensor-noise)
            ColorJitter(p=color_jitter_p),
            GaussianBlur(
                kernel_size=blur_kernel,
                sigma=tuple(blur_sigma),
                p=blur_p),
        ]
        if iso_enabled:
            transforms_list.append(ISONoise(p=iso_p, intensity=iso_intensity))
        transforms_list.extend([
            # Phase 2: Geometric distortions (bbox-aware, with min_visibility
            # filter)
            RandomRotation(degrees=degrees, translate=tuple(translate),
                           min_visibility=min_vis, p=rot_p),
            RandomPerspective(distortion_scale=persp_scale, p=persp_p,
                              min_visibility=min_vis),

            # Phase 3: Spatial sampling (crop with min_visibility=0.5 to drop
            # heavily-cut bboxes)
            RandomCrop(
                scale=tuple(crop_scale),
                p=crop_p,
                min_visibility=min_vis),

            # Phase 4: Resize to model input
            RandomResize([short_edge], max_size=max_size),
            RandomResizeScale(min_scale=scale_min, max_scale=scale_max),

            # Phase 5: Final preprocessing (ToTensor + Normalize, normalized
            # cxcywh boxes)
            normalize,
        ])
        return Compose(transforms_list)

    if image_set == "val":
        short_edge = config.get('val_short_edge', 320) if config else 320
        max_size = config.get('val_max_size', 640) if config else 640
        return Compose([
            RandomResize([short_edge], max_size=max_size),
            normalize,
        ])

    raise ValueError(f"unknown {image_set}")


def build_dataset(image_set, config):
    root = Path(config["data_path"])
    assert root.exists(), f"provided data path {root} does not exist"

    paths = {
        "train": (root / "train", root / config["train_json"]),
        "val": (root / "valid", root / config["valid_json"]),
    }

    img_folder, ann_file = paths[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(
            image_set,
            config))
    return dataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return None
