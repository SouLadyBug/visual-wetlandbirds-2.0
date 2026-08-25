from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = True):
    weights = None
    if pretrained:
        weights = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
            "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        }.get(name)
    constructors = {
        "resnet18": models.resnet18,
        "mobilenet_v2": models.mobilenet_v2,
        "efficientnet_b0": models.efficientnet_b0,
        "densenet121": models.densenet121,
    }
    if name not in constructors:
        raise ValueError(f"Unknown classification model: {name}")
    model = constructors[name](weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name in {"mobilenet_v2", "efficientnet_b0"}:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
