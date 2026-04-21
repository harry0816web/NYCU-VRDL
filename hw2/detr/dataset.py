"""
COCO-format dataset and transforms for DETR.
"""
from pathlib import Path
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

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
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img, target):
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

def make_coco_transforms(image_set):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    scales = list(range(100, 201, 20))  # [100, 120, 140, 160, 180, 200]

    if image_set == "train":
        return Compose([
            ColorJitter(),
            RandomResize(scales),
            normalize,
        ])

    if image_set == "val":
        return Compose([
            RandomResize([200]),
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
        transforms=make_coco_transforms(image_set))
    return dataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return None
