"""All models for project 2, built from ONE shared set of 2D backbones.

Every 2D model can use either backbone, so fusion strategies are compared with
identical feature extractors:

  backbone='scratch'  : the group's 6-layer CNN (3 blocks of conv-BN-ReLU x2,
                        max-pool, dropout), random init           -> 128-d feature
  backbone='resnet18' : ResNet-18 pre-trained on ImageNet           -> 512-d feature

Models (input [B, C, T, H, W] unless noted; output logits [B, num_classes]):
  PerFrame      single frames [B, 3, H, W] in training; on videos the softmax
                is averaged over frames (aggregation of per-frame predictions)
  LateFusion    ONE shared backbone applied to every frame, frame features fused
                (avg: order-invariant, concat: keeps temporal order), then linear
  EarlyFusion   all T frames stacked on the channel axis -> backbone with 3T input
                channels, so the first conv sees every frame
  Simple3DCNN   the group's 3D CNN (3x3x3 convs over T, H, W), scratch only
  TemporalStream  backbone with 2*(T-1) = 18 input channels on stacked flow
"""
import os

import torch
import torch.nn as nn
import torchvision.models as tvm

NUM_CLASSES = 10
NUM_FRAMES = 10


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
def _block2d(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(0.1),
    )


def scratch_backbone(in_channels):
    """The group's CNN (identical to EarlyFusionCNN / _Segment2DCNN) without head."""
    net = nn.Sequential(
        _block2d(in_channels, 32), _block2d(32, 64), _block2d(64, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )
    return net, 128


def resnet18_backbone(in_channels, conv1_init='repeat'):
    """ImageNet ResNet-18 without classifier.

    If in_channels != 3, conv1 is replaced:
      'repeat' (RGB frames stacked as r1,g1,b1,r2,...): the RGB filters are
               copied to every frame and divided by the number of frames.
      'mean'   (optical flow, not colour): the RGB filters are averaged and
               copied to every channel, scaled by 3/in_channels.
    """
    use_weights = os.environ.get('VC_NO_WEIGHTS') != '1'          # only for offline tests
    net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if use_weights else None)
    if in_channels != 3:
        old = net.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            if conv1_init == 'repeat':
                assert in_channels % 3 == 0
                k = in_channels // 3
                new.weight.copy_(old.weight.repeat(1, k, 1, 1) / k)
            else:
                new.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, in_channels, 1, 1)
                                 * 3 / in_channels)
        net.conv1 = new
    net.fc = nn.Identity()
    return net, 512


def make_backbone(name, in_channels=3, conv1_init='repeat'):
    if name == 'scratch':
        return scratch_backbone(in_channels)
    if name == 'resnet18':
        return resnet18_backbone(in_channels, conv1_init)
    raise ValueError(name)


def head(dim, dropout=0.5, num_classes=NUM_CLASSES):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))


def frames_to_batch(x):
    """[B, C, T, H, W] -> [B*T, C, H, W]."""
    B, C, T, H, W = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), B, T


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PerFrame(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone, d = make_backbone(backbone, 3)
        self.head = head(d)

    def forward(self, x):
        if x.dim() == 4:                                    # single frames (training)
            return self.head(self.backbone(x))
        frames, B, T = frames_to_batch(x)                   # whole video (evaluation)
        logits = self.head(self.backbone(frames)).view(B, T, -1)
        return logits.softmax(-1).mean(1).clamp_min(1e-8).log()   # log of avg softmax


class LateFusion(nn.Module):
    def __init__(self, backbone, fusion='avg', num_frames=NUM_FRAMES):
        super().__init__()
        self.backbone, d = make_backbone(backbone, 3)       # ONE shared network for all frames
        self.fusion = fusion
        self.head = head(d * num_frames if fusion == 'concat' else d)

    def forward(self, x):
        frames, B, T = frames_to_batch(x)
        feats = self.backbone(frames).view(B, T, -1)        # [B, T, D]
        fused = feats.mean(1) if self.fusion == 'avg' else feats.flatten(1)
        return self.head(fused)


class EarlyFusion(nn.Module):
    def __init__(self, backbone, num_frames=NUM_FRAMES):
        super().__init__()
        self.backbone, d = make_backbone(backbone, 3 * num_frames, conv1_init='repeat')
        self.head = head(d)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B, T * C, H, W)   # r1,g1,b1,r2,g2,b2,...
        return self.head(self.backbone(x))


def _block3d(cin, cout, pool):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        nn.MaxPool3d(pool), nn.Dropout3d(0.1),
    )


class Simple3DCNN(nn.Module):
    """The group's 3D CNN (same blocks as the 2D scratch CNN, but 3x3x3 kernels)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _block3d(3, 32, (1, 2, 2)),      # keep time
            _block3d(32, 64, (2, 2, 2)),     # T 10 -> 5
            _block3d(64, 128, (1, 2, 2)),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.head = head(128)

    def forward(self, x):
        return self.head(self.features(x))


class TemporalStream(nn.Module):
    def __init__(self, backbone, in_channels=18):
        super().__init__()
        self.backbone, d = make_backbone(backbone, in_channels, conv1_init='mean')
        self.head = head(d)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(model, backbone):
    if model == 'perframe':
        return PerFrame(backbone)
    if model in ('late_avg', 'late_concat'):
        return LateFusion(backbone, model.split('_')[1])
    if model == 'early':
        return EarlyFusion(backbone)
    if model == '3d':
        assert backbone == 'scratch', '3D CNN has no pre-trained version'
        return Simple3DCNN()
    if model == 'temporal':
        return TemporalStream(backbone)
    raise ValueError(model)
