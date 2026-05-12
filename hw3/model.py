"""
model.py - Mask R-CNN with ConvNeXt-Small Backbone + FPN

Architecture:
    Backbone: ConvNeXt-Small (ImageNet-1K pretrained)
    Neck:     Feature Pyramid Network (FPN) with LastLevelMaxPool
    Head:     Standard Mask R-CNN heads (Box + Mask)

Usage:
    from model import get_baseline_model
    model = get_baseline_model(num_classes=5)
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class ConvNeXtFPNBackbone(nn.Module):
    """
    Wraps ConvNeXt-Small feature extractor + FPN into a single backbone
    module compatible with torchvision's MaskRCNN.

    ConvNeXt-Small stage output channels: [96, 192, 384, 768]
    (same as Tiny, but stage 2 has 27 blocks instead of 9)
    FPN output channels: 256 (uniform across all levels)

    Anti-overfitting measures:
        - Stage 0 & 1 frozen (低層特徵已由 ImageNet 學好，209 張圖不需要 finetune)
        - Dropout before FPN (防止 FPN 過度依賴特定 feature channel)
    """

    def __init__(self, out_channels: int = 256, freeze_stages: int = 2,
                 fpn_dropout: float = 0.1):
        super().__init__()
        self.out_channels = out_channels
        self.fpn_dropout_rate = fpn_dropout

        # ── 1. Load ConvNeXt-Small with ImageNet-1K pretrained weights ──
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        convnext = convnext_small(weights=weights)

        # ── 2. Extract features from 4 stages ──
        # ConvNeXt-Small internal node names for each stage output:
        #   features.1 -> after stage 0 (stride 4,  C=96)
        #   features.3 -> after stage 1 (stride 8,  C=192)
        #   features.5 -> after stage 2 (stride 16, C=384)
        #   features.7 -> after stage 3 (stride 32, C=768)
        # Key names MUST be "0","1","2","3" so that MaskRCNN's default
        # MultiScaleRoIAlign (featmap_names=["0","1","2","3"]) can find them.
        return_nodes = {
            "features.1": "0",   # stride 4,  channels=96
            "features.3": "1",   # stride 8,  channels=192
            "features.5": "2",   # stride 16, channels=384
            "features.7": "3",   # stride 32, channels=768
        }
        self.body = create_feature_extractor(convnext, return_nodes=return_nodes)

        # ── 3. Freeze early stages ──
        # ConvNeXt stages: features.0 (stem+stage0), features.1 (downsample),
        #   features.2 (stage1), features.3 (downsample), ...
        # freeze_stages=2 → 凍結 stage 0 和 stage 1 (features.0 ~ features.3)
        if freeze_stages > 0:
            freeze_prefixes = []
            # features.0 = stem + stage 0 blocks
            # features.1 = stage 0→1 downsample
            # features.2 = stage 1 blocks
            # features.3 = stage 1→2 downsample
            # features.4 = stage 2 blocks
            # features.5 = stage 2→3 downsample
            # features.6 = stage 3 blocks
            # features.7 = final LayerNorm
            stage_end_indices = [1, 3, 5, 7]  # 每個 stage 結束的 features.X index
            for s in range(min(freeze_stages, 4)):
                end_idx = stage_end_indices[s]
                if s == 0:
                    start_idx = 0
                else:
                    start_idx = stage_end_indices[s - 1] + 1
                for idx in range(start_idx, end_idx + 1):
                    freeze_prefixes.append(f"body.features.{idx}.")

            frozen_count = 0
            for name, param in self.body.named_parameters():
                if any(name.startswith(prefix) for prefix in freeze_prefixes):
                    param.requires_grad = False
                    frozen_count += 1

            last_frozen = freeze_stages - 1
            print(
                f"[Backbone] Frozen {frozen_count} params in stage 0~{last_frozen}"
            )

        # ── 4. Dropout before FPN ──
        self.fpn_dropout = nn.Dropout2d(p=fpn_dropout) if fpn_dropout > 0 else None

        # ── 5. Build FPN ──
        in_channels_list = [96, 192, 384, 768]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),  # adds P5 (max-pooled) for RPN
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        # Extract multi-scale features from ConvNeXt backbone
        features = self.body(x)

        # Keys are already "0","1","2","3" from create_feature_extractor
        fpn_input = OrderedDict([
            ("0", features["0"]),
            ("1", features["1"]),
            ("2", features["2"]),
            ("3", features["3"]),
        ])

        # Apply dropout before FPN (training only, eval mode auto-disabled)
        if self.fpn_dropout is not None:
            fpn_input = OrderedDict([
                (k, self.fpn_dropout(v)) for k, v in fpn_input.items()
            ])

        # Pass through FPN
        fpn_output = self.fpn(fpn_input)
        return fpn_output


def get_baseline_model(
    num_classes: int = 5,
    min_size=(512,),
    max_size: int = 1024,
    box_detections_per_img: int = 1000,
    rpn_pre_nms_top_n_train: int = 4000,
    rpn_post_nms_top_n_train: int = 3000,
    rpn_pre_nms_top_n_test: int = 3000,
    rpn_post_nms_top_n_test: int = 2000,
) -> MaskRCNN:
    """
    Build a Mask R-CNN model with ConvNeXt-Small + FPN backbone.

    Args:
        num_classes: Number of classes (including background).
                    For this task: 4 cell types + 1 background = 5.
        min_size: Tuple/list of shortest-edge sizes used for multi-scale
                  training resize.
        max_size: Longest-edge cap used during resize.
        box_detections_per_img: Maximum number of detections per image.
                    Default 1000 (torchvision default is 100, too low for
                    dense scenes with 700+ instances).
        rpn_pre_nms_top_n_train: RPN proposals before NMS during training.
        rpn_post_nms_top_n_train: RPN proposals after NMS during training.
        rpn_pre_nms_top_n_test: RPN proposals before NMS during inference.
        rpn_post_nms_top_n_test: RPN proposals after NMS during inference.

    Returns:
        A torchvision MaskRCNN model ready for training.
    """

    # ── Build backbone ──
    backbone = ConvNeXtFPNBackbone(out_channels=256)

    # ── Define anchor generator ──
    # FPN produces 5 feature maps: "0", "1", "2", "3", "pool"
    # We define anchor sizes and aspect ratios for each level.
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )

    # ── Build Mask R-CNN ──
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=min_size,
        max_size=max_size,
        # ── Dense scene settings ──
        box_detections_per_img=box_detections_per_img,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
    )

    return model


if __name__ == "__main__":
    # Quick sanity check
    model = get_baseline_model(num_classes=5)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy_input = [torch.randn(3, 512, 512)]
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output keys: {output[0].keys()}")
    print("Model built successfully!")
