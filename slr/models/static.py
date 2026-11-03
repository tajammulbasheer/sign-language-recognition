"""Static-sign (single image) models: a landmark MLP, and a CNN on raw pixels as a baseline."""

import torch.nn as nn


class LandmarkMLP(nn.Module):
    """Classifies one frame's hand landmarks, (B, J, C) -> logits."""

    def __init__(self, num_joints, channels, num_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_joints * channels, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_cnn(num_classes, arch="mobilenet_v3_large", pretrained=True):
    """ImageNet-pretrained CNN with a new classification layer."""
    import torchvision.models as tvm

    if arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        m = tvm.resnet18(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch {arch}")
    return m
