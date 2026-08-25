from __future__ import annotations

from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def build_detector(num_classes: int, image_size: int, pretrained: bool = True):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = mobilenet_v2(weights=weights).features
    backbone.out_channels = 1280
    anchors = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    return FasterRCNN(backbone, num_classes=num_classes + 1, box_roi_pool=roi, rpn_anchor_generator=anchors, min_size=image_size, max_size=image_size)
