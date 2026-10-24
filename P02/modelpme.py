import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from datasets import FrameVideoDataset

# First we define a 2D CNN with early fusion
class EarlyFusionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.convolutional = nn.Sequential(
             # Block 1: 30 -> 32 channels
            nn.Conv2d(30, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )

        self.fully_connected = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape:
        # [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # [B,30,H,W]
        x = x.reshape(B, C*T, H, W)
        x = self.convolutional(x)
        x = self.fully_connected(x)
        return x


# Translate this to a 3D CNN
# We are basically changing the Conv2d to a Conv3d
# And now we  can also take into account the time dimension of the video
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.convolutional = nn.Sequential(
                        # Block 1: extract low-level spatial and motion features
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Downsample spatial dimensions, preserve time
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.1),

            # Block 2: learn more complex features
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Downsample both time and space
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout3d(0.1),

            # Block 3: learn high-level spatiotemporal features
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Further spatial downsampling only
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.1)
        )

        self.fully_connected = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self,x):
        x = self.convolutional(x)
        x = self.fully_connected(x)
        return x


### Pre trained early fusion model 
class PretrainedEarlyFusionCNN(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()

        # Load ResNet-18 pretrained on ImageNet
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        # Replace the first convolution to accept all RGB frames
        old_conv = self.backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=num_frames * 3,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Initialize the new convolution using pretrained weights
        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.repeat(1, num_frames, 1, 1) / num_frames
            )

        self.backbone.conv1 = new_conv

        # Replace the classifier for your dataset
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

    def forward(self, x):
        # Dataset input: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Rearrange to (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Early fusion: combine all frames into the channel dimension
        x = x.reshape(B, T * C, H, W)

        return self.backbone(x)